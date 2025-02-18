import torch
import torch.nn as nn

import math
import random

from transformers.generation.utils import GenerationMixin
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

import accelerate
from accelerate.logging import get_logger
logger = get_logger('')

from lm_experiments_tools.utils import ObjectView, RMTOutput
    

class GPT2ModelWithBlockWiseMemory(GPT2Model, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config, num_mem_tokens=None):
        super().__init__(config)

        self.config = config
        self.embed_dim = config.hidden_size
        if num_mem_tokens is None:
            self.num_mem_tokens = config.num_mem_tokens
        else:
            self.num_mem_tokens = num_mem_tokens

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        for i in range(config.n_layer):
            self.create_memory(self.num_mem_tokens, i)

        self._gen_indicator = 0
        self._gen_pass_counter = 0

    def create_memory(self, num_mem_tokens, i=None):
        memory_dim = getattr(self.config, 'n_embd', self.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * self.wte.weight.data.std()
        mem_name = f'memory_{i}' if i is not None else 'memory'
        self.register_parameter(mem_name, torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape, i=0):
        memory = getattr(self, f"memory_{i}").repeat(input_shape[0], 1, 1)
        return memory
    
    def pad_attention_mask(self, attention_mask, shape, write_mem=True):
        if self.num_mem_tokens in {0, None}:
            return attention_mask

        mask = torch.zeros(*shape[:2], dtype=self.dtype)[:, None, None, :].to(attention_mask.device)
        if write_mem:
            mask[..., self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[-1]] = attention_mask
        else:
            mask[..., self.num_mem_tokens:] = attention_mask[..., -shape[1] + self.num_mem_tokens:]
        return mask
    
    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.wte(input_ids)
        
        if self.num_mem_tokens > 0:
            if write_mem:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        return (inputs_embeds, self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape), True)
    
    def process_output(self, model_outputs, output_hidden_states=False, output_attentions=False):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
            
            if output_hidden_states:
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if output_attentions:
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs
            
        return out, memory_state

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        memory_states=None,
        labels=None,
        write_mem=False,
        **kwargs,
    ):
        
        # print(input_ids.shape, attention_mask.shape)
        if position_ids is not None:
            position_ids = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        all_memory_states = []
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.num_mem_tokens not in (None, 0):
                if memory_states is None:
                    memory_state = self.set_memory(input_ids.shape, i)
                else:
                    memory_state = memory_states[i]
                
                if write_mem:
                    hidden_states = torch.cat([memory_state, hidden_states, memory_state], dim=1) # batch, mem + len + mem, dim
                elif not self._gen_indicator or self._gen_indicator and self._gen_pass_counter == 0:
                    hidden_states = torch.cat([memory_state, hidden_states], dim=1)

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # print(hidden_states.shape, attention_mask.shape)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if self.num_mem_tokens not in (None, 0):
                if write_mem:
                    memory_state = hidden_states[:, -self.num_mem_tokens:, :]
                    all_memory_states.append(memory_state)
                    hidden_states = hidden_states[:, self.num_mem_tokens:-self.num_mem_tokens, :]
                elif not self._gen_indicator or self._gen_indicator and self._gen_pass_counter == 0:
                    hidden_states = hidden_states[:, self.num_mem_tokens:, :]
            
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        if self._gen_indicator:
            self._gen_pass_counter = 1

        hidden_states = self.ln_f(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        out = RMTOutput(
            # loss=loss,
            logits = lm_logits,
            past_key_values = presents,
            hidden_states = all_hidden_states,
            attentions = all_self_attentions,
            cross_attentions = all_cross_attentions,
            memory_states = all_memory_states
        )
        return out

    def generate(self, *args, **kwargs):
        self._gen_indicator = 1
        self._gen_pass_counter = 0
        out = super().generate(*args, **kwargs)
        self._gen_indicator = 0
        return out



class MemoryAttention(torch.nn.Module):
    def __init__(self, memory_dim, res_mem_count=None, hidden_dim=None, num_heads=1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = memory_dim * 4

        self.memory_dim = memory_dim
        self.res_mem_count = res_mem_count
        self.attention = torch.nn.MultiheadAttention(embed_dim=memory_dim, num_heads=num_heads, batch_first=True)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(memory_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, memory_dim)
        )
        self.norm1 = torch.nn.LayerNorm(memory_dim)
        self.norm2 = torch.nn.LayerNorm(memory_dim)

    def forward(self, current_memory, prev_memories):
        if len(prev_memories) == 0 or self.res_mem_count == 0:
            return current_memory, None
        
        elif self.res_mem_count > 0:
            memory_dropout = random.sample(prev_memories, min(len(prev_memories), self.res_mem_count))
            memory_cat = torch.cat(memory_dropout, dim=1)
            attn_output, attn_map = self.attention(query=current_memory, key=memory_cat, value=memory_cat) # K [memory_dim, n_tokens * n_memory] @ V^T

        else:
            memory_cat = torch.cat(prev_memories, dim=1)
            attn_output, attn_map = self.attention(query=current_memory, key=memory_cat, value=memory_cat)

        updated_memory = current_memory + attn_output 
        updated_memory = self.norm1(updated_memory)
        updated_memory = updated_memory + self.feedforward(updated_memory)
        updated_memory = self.norm2(updated_memory)

        # if return_attention_map:
        return updated_memory, attn_map
        # return updated_memory


def apply_rope_with_segments(memory_state, segment_num):
    batch_size, seq_length, d_model = memory_state.shape

    theta = torch.exp(-1j * (torch.arange(0, d_model, 2, device=memory_state.device) / d_model))
    theta = theta.unsqueeze(0)  #  (1, d_model // 2)
    positions = torch.ones(seq_length, device=memory_state.device, dtype=torch.float) * segment_num
    positions = positions.unsqueeze(-1) # (seq_length, 1)
    complex_pos = torch.exp(1j * positions * theta) # (seq_length, d_model // 2)

    memory_state = memory_state.view(batch_size, seq_length, d_model // 2, 2)
    embeddings_complex = memory_state[..., 0] + 1j * memory_state[..., 1]  # (batch_size, seq_length, d_model // 2)

    embeddings_rotated = embeddings_complex * complex_pos  # (batch_size, seq_length, d_model // 2)

    embeddings_rotated = torch.stack([embeddings_rotated.real, embeddings_rotated.imag], dim=-1)
    return embeddings_rotated.view(batch_size, seq_length, d_model)


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.model = memory_cell
        self.rmt_config = rmt_kwargs

        memory_dim = self.model.memory_0.shape[-1]
        self.n_layer = self.model.config.n_layer
        if rmt_kwargs.get('res_mem_count', None) not in (0, None):
            self.memory_aggregator = nn.ModuleList([MemoryAttention(memory_dim, rmt_kwargs.get('res_mem_count')) for _ in range(self.n_layer)])
        else:
            self.memory_aggregator = None

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=False, output_hidden_states=True):
        memory_states = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        past_memory_states = [[] for _ in range(self.n_layer)]
        memory_attn_outputs = [[] for _ in range(self.n_layer)]
        padded_attn_mask = None
        for seg_num, segment in enumerate(segmented):
            if padded_attn_mask is None or seg_num == len(segmented) - 1:
                padded_attn_mask = self.pad_attention_mask(segment['attention_mask'], segment['input_ids'].shape, write_mem=True)
            segment['attention_mask'] = padded_attn_mask
            cell_out = self.model(**segment, memory_states=memory_states, write_mem=True, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
            cell_outputs.append(cell_out)
            memory_states = cell_out.memory_states

            # apply memory aggregation
            if len(segmented) > 1 and self.memory_aggregator is not None:
                for i in range(self.n_layer):
                    past_memory_states[i].append(apply_rope_with_segments(memory_states[i], seg_num))
                    memory_states[i], memory_attn = self.memory_aggregator[i](memory_states[i], past_memory_states[i])
                    memory_attn_outputs[i].append(memory_attn)
                    memory_states[i] = self.manage_gradients(memory_states[i], seg_num)

        for i in range(len(past_memory_states)):
            past_memory_states[i].clear()
        past_memory_states.clear()


        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   memory_attn_outputs=memory_attn_outputs)
        
        return out
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_states = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        padded_attn_mask = None

        past_memory_states = [[] for _ in range(self.n_layer)]
        for seg_num, segment in enumerate(segmented[:-1]):
            if padded_attn_mask is None:
                padded_attn_mask = self.pad_attention_mask(segment['attention_mask'], segment['input_ids'].shape, write_mem=True)
            segment['attention_mask'] = padded_attn_mask
            out = self.model(**segment, memory_states=memory_states, write_mem=True, output_hidden_states=True)
            memory_states = out.memory_states

            if len(segmented) > 1 and self.memory_aggregator is not None:
                for i in range(self.n_layer):
                    past_memory_states[i].append(apply_rope_with_segments(memory_states[i], seg_num))
                    memory_states[i], _ = self.memory_aggregator[i](memory_states[i], past_memory_states[i])
                    # memory_attn_outputs[i].append(memory_attn)
                    memory_states[i] = self.manage_gradients(memory_states[i], seg_num)

        final_segment = segmented[-1]
        final_segment['attention_mask'] = self.pad_attention_mask(final_segment['attention_mask'], final_segment['input_ids'].shape, write_mem=False)
        out = self.model.generate(**final_segment, memory_states=memory_states, write_mem=False, **generate_kwargs)

        for i in range(len(past_memory_states)):
            past_memory_states[i].clear()
        past_memory_states.clear()

        return out
    
    def pad_attention_mask(self, attention_mask, input_shape, write_mem=True):
        if attention_mask is None:
            return None

        input_shape = list(input_shape)
        num_mem_tokens = self.model.num_mem_tokens

        if num_mem_tokens in [0, None]:
            return attention_mask
        if write_mem:
            input_shape[1] += 2 * num_mem_tokens
        else:
            input_shape[1] += num_mem_tokens
        mask = torch.ones(*input_shape[:2], dtype=torch.int64).to(attention_mask.device)
        mask[:, num_mem_tokens: num_mem_tokens + attention_mask.shape[1]] = attention_mask
        return mask

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = RMTOutput()

        full_logits = torch.cat([o['logits'] for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o['hidden_states'] for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = nn.CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
     
            out['loss'] = loss_fct(flat_logits, flat_labels)
            if out['loss'] is None:
                raise ValueError
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states
        # if len(kwargs.get('memory_attn_outputs', [])) > 0:
        #     out['memory_attn_outputs'] = kwargs.get('memory_attn_outpus')

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]): # and value is not None:
                    out[f'{key}_{seg_num}'] = value

        # if kwargs.get('memory_attn_outputs', False):
        #     return out, kwargs.get('memory_attn_outputs')


        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state

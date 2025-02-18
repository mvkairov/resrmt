import math
import random

import torch
from torch import nn

from lm_experiments_tools.utils import RMTOutput


class MemoryAttention(torch.nn.Module):
    def __init__(self, memory_dim, res_mem_count, hidden_dim=None, num_heads=1):
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
            # if return_attention_map:
            #     return current_memory, None
            return current_memory
        
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
        return updated_memory
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


class MemoryLayerWrapper(nn.Module):
    def __init__(self, layer, num_mem_tokens, memory_dim, res_mem_count=-1, embd_std=0.02):
        super().__init__()
        self.layer = layer

        self.num_mem_tokens = num_mem_tokens
        self.res_mem_count = res_mem_count
        self.create_memory(memory_dim, num_mem_tokens, embd_std)

        self.memory_state = None
        self.aggregator = MemoryAttention(memory_dim=memory_dim, res_mem_count=res_mem_count)
        self.past_memory_states = []

        self.generate_mode = False
        self.first_segment = True
        self.seg_num = 0

    def create_memory(self, memory_dim, num_mem_tokens, embd_std):
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embd_std
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory
    
    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones([shape[0], 1, shape[1], shape[1]], dtype=torch.bool).to(attention_mask.device)
            mask[:, :, self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[-1], self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[-1]] = attention_mask
            return mask

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.seg_num += 1
        if self.memory_state is None:
            self.memory_state = self.set_memory(hidden_states.shape)

        if self.num_mem_tokens > 0:
            if not self.generate_mode:
                # hidden_states = torch.cat([self.memory_state, hidden_states, self.memory_state], dim=1)
                hidden_states[:, :self.num_mem_tokens, :] = self.memory_state
                hidden_states[:, -self.num_mem_tokens:, :] = self.memory_state
            elif self.first_segment:
                # hidden_states = torch.cat([self.memory_state, hidden_states], dim=1)
                hidden_states[:, :self.num_mem_tokens, :] = self.memory_state
        
        # if self.first_segment and attention_mask is not None:
        #     attention_mask = self.pad_attention_mask(attention_mask, hidden_states.shape)
        
        out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)

        if self.num_mem_tokens > 0:
            if not self.generate_mode:
                self.memory_state = apply_rope_with_segments(out[0][:, -self.num_mem_tokens:], self.seg_num)

                if self.aggregator is not None:
                    self.past_memory_states.append(self.memory_state)
                    self.memory_state = self.aggregator(self.memory_state, self.past_memory_states)

                # out = (out[0][:, self.num_mem_tokens:-self.num_mem_tokens, :],) + (out[1:])
            elif self.first_segment:
                self.first_segment = False
                # out = (out[0][:, self.num_mem_tokens:, :],) + (out[1:])

        return out

    def reset_memory(self):
        self.memory_state = None
        self.past_memory_states.clear()
        self.seg_num = 0


class MemoryCell(nn.Module):
    def __init__(self, base_model, num_mem_tokens, res_mem_count=-1, layers_attr: str = 'transformer.h'):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens

        self.layers = self.model
        self.layers_attrs = layers_attr.split('.')
        for i, attr in enumerate(self.layers_attrs):
            self.layers = getattr(self.layers, attr)
        
        for i in range(len(self.layers)):
            self.layers[i] = MemoryLayerWrapper(
                layer=self.layers[i],
                num_mem_tokens=num_mem_tokens,
                memory_dim=self.model.config.hidden_size,
                res_mem_count=res_mem_count,
                embd_std=self.model.get_input_embeddings().weight.data.std()
            )

    def forward(self, input_ids, **kwargs):
        seg_kwargs = self.process_input(input_ids, write_mem=True, **kwargs)
        out = self.model(**seg_kwargs)
        out = self.process_output(out, **kwargs)
        return out
    
    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()
    
    def generate_mode(self, mode):
        for layer in self.layers:
            layer.generate_mode = mode
            layer.first_segment = True
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        self.generate_mode(True)
        seg_kwargs = self.process_input(input_ids, write_mem=False, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        self.generate_mode(False)
        return out

    def process_input(self, input_ids, write_mem=True, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if self.num_mem_tokens > 0:
            blank_tokens = torch.zeros((inputs_embeds.shape[0], self.num_mem_tokens, inputs_embeds.shape[2]), dtype=torch.float, device=inputs_embeds.device)
            if write_mem:
                inputs_embeds = torch.cat([blank_tokens, inputs_embeds, blank_tokens], dim=1)
            else:
                inputs_embeds = torch.cat([blank_tokens, inputs_embeds], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape,
                                                                   kwargs.get('write_mem', True))
            # seg_kwargs['attention_mask'] = kwargs.get('attention_mask')
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = RMTOutput()
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
            
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            out = model_outputs
        
        return out 
    
    def pad_attention_mask(self, attention_mask, shape, write_mem):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            shape = list(shape)[:2]
            # shape[1] += self.num_mem_tokens * (1 + write_mem)
            mask = torch.ones(*shape, dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[1]] = attention_mask
            return mask
        
    def manage_gradients(self, *args, **kwargs):
        for layer in self.layers:
            layer.memory_state = self._manage_gradients(layer.memory_state, *args, **kwargs)
        
    def _manage_gradients(self, memory_state, seg_num, k2, max_n_segments):
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state


class RecurrentWrapper(nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        self.memory_cell.reset_memory()
        for seg_num, segment in enumerate(segmented):
            print(seg_num, segment['input_ids'].shape)
            cell_out = self.memory_cell(**segment, output_hidden_states=True)
            cell_outputs.append(cell_out)
            self.memory_cell.manage_gradients(seg_num, self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments'))

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        self.memory_cell.reset_memory()
        for segment in segmented[:-1]:
            self.memory_cell(**segment)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, **generate_kwargs)

        return out

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
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

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

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        return out 

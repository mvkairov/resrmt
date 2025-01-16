#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_babilong_baseline.sh
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cd ../..
CUDA_VISIBLE_DEVICES=0
NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
BACKBONE_CLS=transformers:AutoModelForCausalLM
NOISE_DATASET=pg19
METRIC=exact_match
POSTFIX=teacher-forcing

# for MODEL_KIND in rmt resrmt; do
for MODEL_KIND in bwrmt; do

if [ $MODEL_KIND = "rmt" ]; then
    MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
elif [ $MODEL_KIND = "resrmt" ]; then
    MEMORY_CELL=modeling_rmt.resrmt:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.resrmt:RecurrentWrapper
elif [ $MODEL_KIND = "bwrmt" ]; then
    BACKBONE_CLS=modeling_rmt.block_resrmt:GPT2ModelWithBlockWiseMemory
    MEMORY_CELL="none --no_memory_cell"
    RECURRENT_WRAPPER=modeling_rmt.block_resrmt:RecurrentWrapper
else
    exit 1
fi

MODEL_NAME=gpt2  # backbone model
    
ITERS=10000

# for TASK_DATASET in qa1_single-supporting-fact qa4_two-arg-relations; do
for TASK_DATASET in qa3_three-supporting-facts; do
# for TASK_DATASET in qa4_two-arg-relations

for LR in 1e-05; do

TBS=6
for SEGMENT_SIZE in 512; do
MAX_N_SEGMENTSS=(0 0 1 0 0 3 0 0 6)
BSS=(0 0 6 0 0 2 0 0 1)

# MAX_N_SEGMENTSS=(0 0 1)
# BSS=(0 0 6)

for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ )); do

MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=${BSS[j]}

j1=$((j-1))
SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}

j2=$((j-2))
SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}

if [ $MAX_N_SEGMENTS -ne 0 ]; then

for MEMORY_SIZE in 16; do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens

GRAD_ACC_STEPS=$(($TBS/$BS))

SCHEDULER=linear

for RES_MEM_COUNT in -1; do

for N in test2; do

K2=-1 # BPTT unroll length

NP=$NP
ACCEL_CONFIG=/data/home/admin/rmt/accel_configs/exp/accelerate/${MODEL_KIND}_bf16_tbs${TBS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0\
        --prefix $MODEL_KIND
cd ..

MODEL_PATH="/data/home/admin/rmt/runs/${TASK_DATASET}/${MODEL_NAME}/${MODEL_KIND}/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}_${POSTFIX}/run_${N}"

if [ ! -d $MODEL_PATH ]; then

echo RUNNING: MODEL_KIND $MODEL_KIND TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE RES_MEM_COUNT $RES_MEM_COUNT SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

MODEL_CPT="/data/home/admin/rmt/runs/${TASK_DATASET}/${MODEL_NAME}/${MODEL_KIND}/${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}_${POSTFIX}/run_${N}/model_best"

if [ ! -d $MODEL_CPT ]; then
    echo checkpoint not found, training from scratch
    MODEL_CPT=""
else
    echo checkpoint found
    MODEL_CPT="--model_cpt ${MODEL_CPT}"
fi

accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29007 run_finetuning_babilong_resrmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /data/home/admin/rmt/data/tasks_1-20_v1-2/en-10k \
        --model_path $MODEL_PATH $MODEL_CPT \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --residual_memory_count $RES_MEM_COUNT \
        --tokenizer gpt2 \
        --vary_n_segments \
        --batch_size $BS  \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --reset_optimizer --reset_lr --reset_iteration \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS / 10)) \
        --data_n_workers 2 \
        --log_interval 25 --valid_interval 100 \
        --optimize_metric $METRIC --optimize_mode max --best_metric_value 1.0 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0
        # --validate_only

else

echo run $MODEL_PATH exists already, delete the previous run or change its name

fi

done
done
done

fi

done
done
done
done
done

echo done

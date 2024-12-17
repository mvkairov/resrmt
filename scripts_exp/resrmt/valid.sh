#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
export TOKENIZERS_PARALLELISM=false
# cd ../..
CUDA_VISIBLE_DEVICES=0
NP=4
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.resrmt:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.resrmt:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM

NOISE_DATASET=pg19
METRIC=exact_match

MODEL_KIND=resrmt
POSTFIX=teacher-forcing
MODEL_NAME=gpt2  # backbone model

ITERS=10000
TBS=6

# for TASK_DATASET in qa1_single-supporting-fact
# for TASK_DATASET in qa3_three-supporting-facts
for TASK_DATASET in qa4_two-arg-relations
do

for LR in 1e-05
do

for SEGMENT_SIZE in 512
do

# MAX_N_SEGMENTSS=(0 0 1 2 4 6)

for MAX_N_SEGMENTS in 45
do

SRC_N_SEGMENTSS=(3 6)

for (( j=1; j<${#SRC_N_SEGMENTSS[@]}; j++ ))
do

SRC_N_SEGMENTS=${SRC_N_SEGMENTSS[j]}

j1=$((j-1))
SRC_SRC_N_SEGMENTS=${SRC_N_SEGMENTSS[j1]}
echo SRC_N_SEGMENTS $SRC_N_SEGMENTS SRC_SRC_N_SEGMENTS $SRC_SRC_N_SEGMENTS

for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens

GRAD_ACC_STEPS=1

SCHEDULER=linear

for RES_MEM_COUNT in -1
do

for N in 0
do

K2=-1   # BPTT unroll length

NP=$NP
ACCEL_CONFIG=/data/home/admin/rmt/accel_configs/exp/accelerate/${MODEL_KIND}_bf16_tbs${TBS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $TBS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0\
        --prefix $MODEL_KIND
cd ..

echo RUNNING: MODEL_KIND $MODEL_KIND TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE RES_MEM_COUNT $RES_MEM_COUNT SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

MODEL_PATH="/data/home/admin/rmt/runs/${TASK_DATASET}/${MODEL_NAME}/${MODEL_KIND}/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}_${POSTFIX}/run_${N}"
MODEL_CPT="/data/home/admin/rmt/runs/${TASK_DATASET}/${MODEL_NAME}/${MODEL_KIND}/${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}_${POSTFIX}/run_${N}/model_best"
if [ ! -d "$MODEL_CPT" ]; then
MODEL_CPT=""
else
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
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --reset_optimizer --reset_lr --reset_iteration \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS / 10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS / 100)) --valid_interval $(($ITERS / 25)) \
        --optimize_metric $METRIC --optimize_mode max --best_metric_value 1.0 \
        --show_valid_examples 5 \
        --early_stopping_patience 10 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0 \
        --validate_only

done
done
done
done
done
done
done
done
echo skibidi toilet

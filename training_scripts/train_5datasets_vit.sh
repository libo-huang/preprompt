#!/bin/bash
for seed in 1024
do
LOGDIR="./output/5datasets_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port=29501 main.py \
        five_datasets_preprompt_5e \
        --original_model vit_base_patch16_224 \
        --model vit_base_patch16_224 \
        --batch-size 32 \
        --data-path ./datasets \
        --output_dir $LOGDIR \
        --epochs 20 \
        --sched constant \
        --seed $seed \
        --train_inference_task_only \
        --lr 0.01 \
        2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done

for seed in 1024
do
LOGDIR="./output/5datasets_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port=29511 main.py \
        five_datasets_preprompt_5e \
        --original_model vit_base_patch16_224 \
        --model vit_base_patch16_224 \
        --batch-size 32 \
        --data-path ./datasets \
        --output_dir $LOGDIR \
        --epochs 20 \
        --sched constant \
        --lr 0.01 \
        --clip-grad 2 \
        --reg 0 \
        --prompt_momentum 0.01 \
        --seed $seed \
        --larger_prompt_lr \
        --trained_original_model ./output/5datasets_vit_seed${seed} \
        2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done
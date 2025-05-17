#!/bin/bash
for seed in 1024
do
LOGDIR="./output/imr_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port='29500' main.py \
        imr_preprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ${LOGDIR} \
        2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done

for seed in 1024
do
LOGDIR="./output/imr_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port='29505' main.py \
        imr_preprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 150 \
        --data-path ./datasets \
        --ca_lr 0.01 \
        --lr 0.01 \
        --crct_epochs 30 \
        --sched cosine \
        --seed $seed \
        --prompt_momentum 0.01 \
        --reg 0.0 \
        --length 20 \
        --larger_prompt_lr \
        --trained_original_model ./output/imr_vit_seed${seed} \
        --output_dir ${LOGDIR} \
        2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done
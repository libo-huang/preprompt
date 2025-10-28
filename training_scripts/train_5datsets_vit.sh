#!/bin/bash

for seed in 42  # 40 44
do
LOGDIR="./output/5datasets_vit_pe_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port=29501 main.py \
        five_datasets_preprompt_5e \
        --model vit_base_patch16_224 \
        --batch-size 32 \
        --epochs 20 \
        --data-path ./datasets \
        --ca_lr 0.07 \
        --crct_epochs 30 \
        --num_tasks 5 \
        --seed $seed \
        --prompt_momentum 0.01 \
        --length 5 \
        --sched constant \
        --larger_prompt_lr \
        --clip-grad 2 \
        --output_dir $LOGDIR \
        --ca_storage_efficient_method feat-trans \
        2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done
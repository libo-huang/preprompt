#!/bin/bash

for seed in 42 # 40 44
do
LOGDIR="./output/imr_vit_pe_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port=29501 main.py \
        imr_preprompt_5e \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 150 \
        --data-path ./datasets \
        --ca_lr 0.003 \
        --crct_epochs 30 \
        --num_tasks 10 \
        --seed $seed \
        --prompt_momentum 0.01 \
        --length 10 \
        --sched cosine \
        --larger_prompt_lr \
        --output_dir ${LOGDIR} \
	--ca_storage_efficient_method feat-trans \
        2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done 
#!/bin/bash

for seed in 42
do
LOGDIR="./output/cub_vit_pe_2_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port=29520 main.py \
	cub_preprompt_5e \
	--model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 50 \
	--data-path ./datasets \
	--ca_lr 0.01 \
	--crct_epochs 30 \
	--num_tasks 10 \
	--seed $seed \
	--prompt_momentum 0.01 \
	--length 20 \
	--output_dir ${LOGDIR} \
	--ca_storage_efficient_method feat-trans \
	2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done
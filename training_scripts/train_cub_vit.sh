#!/bin/bash
for seed in 1024
do
LOGDIR="./output/cub_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port='29520' main.py \
	cub_preprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 20 \
	--data-path ./datasets \
	--lr 0.01 \
	--ca_lr 0.01 \
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
LOGDIR="./output/cub_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port='29502' main.py \
	cub_preprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 50 \
	--data-path ./datasets \
	--ca_lr 0.01 \
	--crct_epochs 30 \
	--seed $seed \
	--prompt_momentum 0.01 \
	--length 20 \
	--trained_original_model ./output/cub_vit_seed${seed} \
	--output_dir $LOGDIR \
	2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done
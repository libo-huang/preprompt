#!/bin/bash
for seed in 1024
do
LOGDIR="./output/cifar100_sup21k_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 --master_port=29501 main.py \
		cifar100_preprompt_5e \
		--original_model vit_base_patch16_224 \
		--model vit_base_patch16_224 \
		--batch-size 24 \
		--data-path ./datasets/ \
		--seed $seed \
		--epochs 1 \
		--sched constant \
		--output_dir $LOGDIR \
		--lr 0.01 \
		--train_inference_task_only \
		2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done

		# --epochs 20 \


for seed in 1024
do
LOGDIR="./output/cifar100_vit_seed${seed}"
LOGFILE=$(date +"${LOGDIR}/log_%Y_%m_%d_%H_%M_output.log")
mkdir -p "$LOGDIR"
torchrun --nproc_per_node=8 main.py \
	cifar100_preprompt_5e \
	--original_model vit_base_patch16_224 \
	--model vit_base_patch16_224 \
	--batch-size 24 \
	--data-path ./datasets \
	--seed $seed \
	--epochs 1 \
	--sched step \
	--output_dir ${LOGDIR} \
	--lr 0.01 \
	--ca_lr 0.01 \
	--crct_epochs 1 \
	--prompt_momentum 0.01 \
	--length 5 \
	--larger_prompt_lr \
	--trained_original_model ./output/cifar100_sup21k_seed${seed} \
	2>&1 | tee "$LOGFILE"
echo "LOGDIR: $LOGDIR"
echo "LOGFILE: $LOGFILE"
done
	# --epochs 50 \
	# --crct_epochs 30 \
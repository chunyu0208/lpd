OUTPUT_DIR="output/lpd_xxl_256"
CACHED_PATH="path/to/your_256_cached_latents"
WANDB_RUN_NAME="lpd-xxl-256"

mkdir -p ${OUTPUT_DIR}

source scripts/setups/train.sh

export WANDB_RESUME="allow"

torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    main.py \
    --img_size 256 --vqgan_path tokenizers/vq_ds16_c2i.pt \
    --use_cached --cached_path ${CACHED_PATH} \
    --model lpd_xxl \
    --epochs 450 --warmup_epochs 50 --lr_decay_start_epoch 400 \
    --batch_size 32 --bf16 \
    --lr_schedule "constant+cosine" --blr 1.0e-4 --min_lr 1.0e-5 \
    --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
    --group_sizes_list orders/group_sizes_training/group_sizes_256.json \
    --online_eval --eval_freq 10 --group_sizes "[1]*256" \
    --cfg 4.0 --cfg_schedule "linear" \
    --save_intermediate_freq 25 --save_last_freq 5 \
    --use_wandb --wandb_run_name ${WANDB_RUN_NAME}
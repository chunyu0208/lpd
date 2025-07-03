OUTPUT_DIR="output/lpd_xl_512"
CACHED_PATH="path/to/your_512_cached_latents"
WANDB_RUN_NAME="lpd-xl-512"

mkdir -p ${OUTPUT_DIR}

source scripts/setups/train.sh

export WANDB_RESUME="allow"

torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    main.py \
    --img_size 512 --vqgan_path tokenizers/vq_ds16_c2i.pt \
    --use_cached --cached_path ${CACHED_PATH} \
    --model lpd_xl \
    --pretrained_ckpt path/to/pretrained_lpd_xl_256.pth \
    --epochs 50 --warmup_epochs 1 \
    --batch_size 8 --bf16 \
    --lr_schedule "cosine" --blr 1.0e-4 --min_lr 1.0e-5 \
    --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
    --group_sizes_list orders/group_sizes_training/group_sizes_512.json \
    --online_eval --eval_freq 10 --group_sizes "[1]*1024" \
    --cfg 4.0 --cfg_schedule "linear" \
    --save_intermediate_freq 25 --save_last_freq 5 \
    --use_wandb --wandb_run_name ${WANDB_RUN_NAME}
torchrun --nproc_per_node=8 --nnodes=1 \
    main.py \
    --img_size 256 \
    --vqgan_path tokenizers/vq_ds16_c2i.pt \
    --model lpd_xl \
    --pretrained_ckpt path/to/your_downloaded_model/lpd_xl_256.pth \
    --evaluate \
    --eval_bsz 256 \
    --bf16 \
    --generation_steps 32 \
    --order lpd \
    --lpd_order_file orders/lpd_orders_generated/res256_steps32_repulsion2.json \
    --cfg 5.0 \
    --cfg_schedule "linear" \
    --output_dir path/to/your_output_dir
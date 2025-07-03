torchrun --nproc_per_node=8 --nnodes=1 \
    main.py \
    --img_size 512 \
    --vqgan_path tokenizers/vq_ds16_c2i.pt \
    --model lpd_xl \
    --pretrained_ckpt path/to/your_downloaded_model/lpd_xl_512.pth \
    --evaluate \
    --eval_bsz 64 \
    --bf16 \
    --generation_steps 48 \
    --order lpd \
    --lpd_order_file orders/lpd_orders_generated/res512_steps48_repulsion4.json \
    --cfg 6.0 \
    --cfg_schedule "linear" \
    --output_dir path/to/your_output_dir
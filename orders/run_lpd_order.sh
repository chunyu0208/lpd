# 256x256, 20 steps, repulsion threshold 2.0
echo "Generating lpd orders for resolution 256x256, generation steps 20, and repulsion threshold 2.0"
python orders/run_lpd_order.py \
    --img_size 256 \
    --generation_steps 20 \
    --repulsion_threshold 2.0 \
    --output_file orders/lpd_orders_generated/res256_steps20_repulsion2.json

# 256x256, 20 steps, repulsion threshold 3.0
echo "Generating lpd orders for resolution 256x256, generation steps 20, and repulsion threshold 3.0"
python orders/run_lpd_order.py \
    --img_size 256 \
    --generation_steps 20 \
    --repulsion_threshold 3.0 \
    --output_file orders/lpd_orders_generated/res256_steps20_repulsion3.json

# 256x256, 32 steps, repulsion threshold 2.0
echo "Generating lpd orders for resolution 256x256, generation steps 32, and repulsion threshold 2.0"
python orders/run_lpd_order.py \
    --img_size 256 \
    --generation_steps 32 \
    --repulsion_threshold 2.0 \
    --output_file orders/lpd_orders_generated/res256_steps32_repulsion2.json

# 512x512, 48 steps, repulsion threshold 4.0
echo "Generating lpd orders for resolution 512x512, generation steps 48, and repulsion threshold 4.0"
python orders/run_lpd_order.py \
    --img_size 512 \
    --generation_steps 48 \
    --repulsion_threshold 4.0 \
    --output_file orders/lpd_orders_generated/res512_steps48_repulsion4.json
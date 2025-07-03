import argparse
import json
import os

from lpd_order import run_lpd_order_schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--generation_steps', type=int, default=None)
    parser.add_argument("--group_sizes", type=str, default=None)
    parser.add_argument("--proximity_threshold", type=float, default=1.0)
    parser.add_argument("--repulsion_threshold", type=float, default=1.0)
    parser.add_argument("--output_file", type=str, default=None)

    parser.add_argument("--num_runs", type=int, default=60000)
    parser.add_argument("--max_workers", type=int, default=128)
    args = parser.parse_args()

    if args.generation_steps is not None:
        with open("orders/group_sizes_lookup.json", "r") as f:
            group_sizes_lookup = json.load(f)
        try:
            args.group_sizes = group_sizes_lookup[f"{args.img_size} resolution"][f"{args.generation_steps} steps"]
        except KeyError:
            raise ValueError(f"Group sizes for {args.img_size} resolution and {args.generation_steps} steps not found in group_sizes_lookup.json")
    
    if args.group_sizes is None:
        raise ValueError("Please provide either --generation_steps or --group_sizes")
    
    if args.img_size == 256:
        grid_size = 16
    elif args.img_size == 512:
        grid_size = 32
    else:
        raise ValueError(f"Image size {args.img_size} not supported")
    
    # Generate orders in batches to manage memory efficiently
    batch_size = 5000
    num_batches = args.num_runs // batch_size
    total_orders = []
    
    print(f"Generating {args.num_runs} orders in {num_batches} batches of {batch_size}")
    
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        batch_results = run_lpd_order_schedule(
            num_runs=batch_size, 
            group_sizes=eval(args.group_sizes) if isinstance(args.group_sizes, str) else args.group_sizes,
            grid_size=grid_size,
            proximity_threshold=args.proximity_threshold,
            repulsion_threshold=args.repulsion_threshold,
            max_workers=args.max_workers
        )
        total_orders.extend(batch_results)
    
    # Save results to output file
    print(f"Saving {len(total_orders)} orders to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(total_orders, f)
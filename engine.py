import argparse
import cv2
import json
import math
import numpy as np
import os
import random
import shutil
import sys
import time
import torch
import torch_fidelity
import wandb

from typing import Iterable

from models.lpd import LPD
from models.vqgan import VQModel
from util import misc, lr_sched


def create_block_causal_mask(block_sizes, device=None, dtype=torch.float32):
    total_len = sum(block_sizes)
    mask = torch.full((total_len, total_len), float('-inf'), dtype=dtype, device=device)

    start = 0
    for size in block_sizes:
        end = start + size
        mask[start:end, :end] = 0
        start = end

    return mask

def create_evaluation_orders(model_without_ddp: LPD, order: str = 'random', total_samples: int = 50000, lpd_order_file: str = None):
    if order == 'random':
        orders = model_without_ddp.sample_orders(bsz=total_samples)
    elif order == 'lpd':
        with open(lpd_order_file, "r") as f:
            orders = json.load(f)
        orders = torch.tensor(orders[:total_samples], device='cuda')
    else:
        raise NotImplementedError

    return orders

def train_one_epoch(
        model, vqgan: VQModel,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler,
        log_writer=None,
        args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_cached:
                x = samples
            else:
                latent, _, [_, _, indices] = vqgan.encode(samples)
                x = indices.reshape(latent.shape[0], -1)

        # forward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else None):
            if args.group_sizes_list is not None:
                with open(args.group_sizes_list, 'r') as f:
                    group_sizes_list = json.load(f)
                group_sizes = random.choice(group_sizes_list)
            else:
                group_sizes = args.group_sizes
            loss = model(x, labels, group_sizes)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            # Log to wandb if enabled
            if args.use_wandb and misc.is_main_process():
                wandb.log({
                    'train_loss': loss_value_reduce,
                    'lr': lr,
                    'epoch': epoch + (data_iter_step / len(data_loader)),
                    'epoch_1000x': epoch_1000x,
                }, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp: LPD, 
             vqgan: VQModel, 
             args: argparse.Namespace, 
             epoch: int, 
             batch_size: int = 16, 
             log_writer=None, 
             cfg: float = 1.0):
    
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "temp{}-{}cfg{}-topk{}-topp{}-image{}".format(args.temperature,
                                                                            args.cfg_schedule,
                                                                            cfg,
                                                                            args.top_k,
                                                                            args.top_p,
                                                                            args.num_images))

    save_folder = save_folder + "_evaluate"

    print("Save to:", save_folder)

    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    if args.generation_steps is not None:
        with open("orders/group_sizes_lookup.json", "r") as f:
            group_sizes_lookup = json.load(f)
        try:
            args.group_sizes = group_sizes_lookup[f"{args.img_size} resolution"][f"{args.generation_steps} steps"]
        except KeyError:
            raise ValueError(f"Group sizes for {args.img_size} resolution and {args.generation_steps} steps not found in group_sizes_lookup.json")

    # Create attention mask
    block_sizes = [1]
    block_sizes.extend(eval(args.group_sizes) if isinstance(args.group_sizes, str) else args.group_sizes)
    attn_mask = create_block_causal_mask(block_sizes, device = "cuda")

    # Create orders
    all_eval_orders = create_evaluation_orders(model_without_ddp, args.order, num_steps*batch_size*world_size, args.lpd_order_file)

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        eval_orders = all_eval_orders[world_size * batch_size * i + local_rank * batch_size:
                                        world_size * batch_size * i + (local_rank + 1) * batch_size]

        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else None):
                sampled_tokens = model_without_ddp.sample_tokens(
                    bsz=batch_size, 
                    cfg=cfg,
                    cfg_schedule=args.cfg_schedule, 
                    labels=labels_gen,
                    temperature=args.temperature, 
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    group_sizes=args.group_sizes, 
                    attn_mask=attn_mask, 
                    orders=eval_orders
                )
                
                sampled_images = vqgan.decode_code(
                    sampled_tokens, 
                    shape = (
                        sampled_tokens.shape[0], 
                        -1, 
                        int(sampled_tokens.shape[1]**0.5), 
                        int(sampled_tokens.shape[1]**0.5)
                        )
                    )

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.to(torch.float32)

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        elif args.img_size == 512:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        
        # Log to wandb if enabled
        if args.use_wandb and misc.is_main_process():
            wandb.log({
                'fid{}'.format(postfix): fid,
                'is{}'.format(postfix): inception_score,
            }, step=(epoch+1) * 1000)
            
            # Log sample images to wandb (optional)
            try:
                # Get a few sample images from the generated folder
                sample_images = []
                image_files = [f for f in os.listdir(save_folder) if f.endswith('.png') or f.endswith('.jpg')][:64]
                for img_file in image_files:
                    img_path = os.path.join(save_folder, img_file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    sample_images.append(wandb.Image(img, caption=img_file))
                
                wandb.log({
                    "generated_samples{}".format(postfix): sample_images,
                }, step=(epoch+1) * 1000)
            except Exception as e:
                print(f"Warning: Could not log sample images to wandb: {e}")
        
        # remove temporal saving folder
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)

def cache_latents(
        vqgan: VQModel,
        data_loader: Iterable,
        device: torch.device,
        args=None
    ):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        B, C, H, W = samples.shape

        with torch.no_grad():
            _, _, [_, _, indices] = vqgan.encode(samples)
            indices = indices.view(B, -1)
            _, _, [_, _, indices_flip] = vqgan.encode(samples.flip(dims=[3]))
            indices_flip = indices_flip.view(B, -1)

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, indices=indices[i].cpu().numpy(), indices_flip=indices_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return

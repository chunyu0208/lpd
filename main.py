import argparse
import datetime
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from engine import evaluate, train_one_epoch
from models import lpd
from models.vqgan import VQModel
from util import misc
from util.crop import center_crop_arr
from util.loader import CachedFolderVQ
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Locality-aware Parallelized Decoding Training', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='lpd_l', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vqgan_path', default="tokenizers/vq_ds16_c2i.pt", type=str,
                        help='vqgan path')
    parser.add_argument('--vqgan_stride', default=16, type=int,
                        help='tokenizer stride, default 16')
    parser.add_argument('--vqgan_vocab_size', default=16384, type=int, help='vqgan vocab size')
    parser.add_argument('--pretrained_ckpt', default=None, type=str, help='pretrained checkpoint')

    # Training parameters
    parser.add_argument('--group_sizes_list', type=str, default=None, help='group_sizes list json file')

    parser.add_argument('--epochs', default=450, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant+cosine',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--lr_decay_start_epoch', type=int, default=400, metavar='N',
                        help='epochs to start LR decay')
    parser.add_argument('--bf16', action='store_true', help='use bf16')

    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last checkpoint frequency')
    parser.add_argument('--save_intermediate_freq', type=int, default=25, help='save intermediate checkpoint frequency')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume_epoch', default=0, type=int, metavar='N',
                        help='resume epoch')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--lpd_order_file', default=None, type=str)

    # Sampling parameters
    parser.add_argument('--generation_steps', type=int, default=None, help='generation steps')
    parser.add_argument('--group_sizes', type=str, default='[1] * 256', help='group_sizes')
    parser.add_argument('--order', type=str, default='random', help='inference order')
    parser.add_argument('--temperature', default=1.0, type=float, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='top-k sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p sampling')

    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='whether to use wandb for logging')
    parser.add_argument('--wandb_project', default='ar-cache', type=str, help='wandb project name')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='wandb run name')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def interpolate_pos_embed(state_dict, model, mode = 'bicubic'):
    pos_embed_checkpoint = state_dict['pos_embed']

    num_tokens_old = state_dict['pos_embed'].shape[1] - 1
    num_tokens_new = model.pos_embed.shape[1] - 1

    grid_size_old = int(num_tokens_old ** 0.5)
    grid_size_new = int(num_tokens_new ** 0.5)

    pos_embed = pos_embed_checkpoint[0, 1:].reshape(1, grid_size_old, grid_size_old, -1).permute(0, 3, 1, 2)
    pos_embed_interpolated = torch.nn.functional.interpolate(
        pos_embed, size = (grid_size_new, grid_size_new), mode = mode
    )

    pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1).reshape(1, num_tokens_new, -1)

    pos_embed_new = torch.cat(
        [pos_embed_checkpoint[:, :1],
         pos_embed_interpolated],
         dim = 1)

    return pos_embed_new


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=f"run-id-{args.wandb_run_name}",
            config=args,
            dir=args.output_dir,
            resume="allow"
        )

    transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.evaluate is False:
        if args.use_cached:
            dataset_train = CachedFolderVQ(args.cached_path)
        else:
            dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    vqgan = VQModel(codebook_size=args.vqgan_vocab_size).cuda().eval()

    vqgan_ckpt = torch.load(args.vqgan_path, map_location="cpu")
    if 'model' in vqgan_ckpt:
        state_dict = vqgan_ckpt['model']
    else:
        state_dict = vqgan_ckpt
    vqgan.load_state_dict(state_dict)

    for param in vqgan.parameters():
        param.requires_grad = False

    model = lpd.__dict__[args.model](
        img_size=args.img_size,
        vqgan_stride=args.vqgan_stride,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        grad_checkpointing=args.grad_checkpointing,
        vqgan_vocab_size=args.vqgan_vocab_size
    )

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.resume_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    elif args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        if 'pos_embed' in state_dict:
            if model_without_ddp.pos_embed.shape != state_dict['pos_embed'].shape:
                state_dict['pos_embed'] = interpolate_pos_embed(state_dict, model_without_ddp)
        model_without_ddp.load_state_dict(state_dict)
        print("Load pretrained checkpoint %s" % args.pretrained_ckpt)
    else:
        print("Training from scratch")

    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vqgan, args, 0, batch_size=args.eval_bsz, log_writer=log_writer, cfg=args.cfg)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.resume_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vqgan,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if (epoch+1) % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, epoch=epoch, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch_name="last")
        
        if (epoch+1) % args.save_intermediate_freq == 0:
            misc.save_model(args=args, epoch=epoch, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch_name=f"{epoch+1}")

        if args.online_eval and ((epoch+1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vqgan, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                     cfg=1.0)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vqgan, args, epoch, batch_size=args.eval_bsz // 2,
                         log_writer=log_writer, cfg=args.cfg)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.use_wandb and misc.is_main_process():
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)

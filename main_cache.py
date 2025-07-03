import argparse
import datetime
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from engine import cache_latents
from models.vqgan import VQModel
from util import misc
from util.crop import center_crop_arr
from util.loader import ImageFolderWithFilename


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VQGAN latent codes', add_help=False)
    # Cache parameters
    parser.add_argument('--vqgan_path', default="tokenizers/vq_ds16_c2i.pt", type=str,
                        help='vqgan path')
    parser.add_argument('--vqgan_vocab_size', default=16384, type=int, help='vqgan vocab size')
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)  
    
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


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

    transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    vqgan = VQModel(codebook_size=args.vqgan_vocab_size).cuda().eval()

    ckpt = torch.load(args.vqgan_path, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    vqgan.load_state_dict(state_dict)

    print(f"Start caching VQGAN latent codes")
    start_time = time.time()
    cache_latents(
        vqgan,
        data_loader_train,
        device,
        args=args
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
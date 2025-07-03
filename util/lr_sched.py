import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.lr_schedule == "constant+cosine":
            if epoch <= args.lr_decay_start_epoch:
                lr = args.lr
            else:
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - args.lr_decay_start_epoch) / (args.epochs - args.lr_decay_start_epoch)))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

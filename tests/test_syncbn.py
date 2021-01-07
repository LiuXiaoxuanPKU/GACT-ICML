import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from quantize import QSyncBatchNorm, config

N = 16

if __name__ == '__main__':
    config.activation_compression_bits = 8
    config.pergroup = False
    config.perlayer = False

    parser = argparse.ArgumentParser(description='Test SyncBN')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--quantize", default=0, type=int)
    args = parser.parse_args()

    print('Process ', args.local_rank)

    # Initialize distributed
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        print('Initializing {}'.format(args.gpu))
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        print('Initialized {} / {}'.format(args.gpu, args.world_size))
    else:
        args.gpu = 0
        args.world_size = 1

    local_N = N // args.world_size

    if args.distributed:
        if args.quantize == 0:
            model = nn.SyncBatchNorm(10).cuda()
        else:
            model = QSyncBatchNorm(10).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = nn.BatchNorm2d(10).cuda()

    model.train()

    data = torch.load('syncbn.pt')
    if args.distributed:
        m = model.module
    else:
        m = model

    with torch.no_grad():
        m.weight.copy_(data['w'].cuda())
        m.bias.copy_(data['b'].cuda())

    x = data['x'].view(N, 10, 1, 1)
    my_x = x[args.gpu*local_N : (args.gpu+1)*local_N]
    my_x = my_x.cuda()
    my_x.requires_grad_()

    output = model(my_x)
    loss = (output**2).mean()
    loss.backward()

    for i in range(args.world_size):
        if i == args.gpu:
            print(my_x.grad.squeeze(), m.weight, m.bias, m.weight.grad, m.bias.grad)
        if args.distributed:
            dist.barrier()

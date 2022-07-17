"""
Modified from https://github.com/utsaslab/MONeT/blob/master/examples/imagenet.py
"""

import argparse
import json
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

from scaled_resnet import scaled_resnet, scaled_wide_resnet
import gact
from gact import config
from gact.utils import get_memory_usage, compute_tensor_bytes, exp_recorder

MB = 1024**2
GB = 1024**3
MEM_LIMIT = 15 * GB

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    # choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=2, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ablation', action="store_true",
                    help="Do ablation?")
parser.add_argument('--benchmark', type=str, default='exact')
parser.add_argument('--limit', type=float, default=1, help="memory percentage")
parser.add_argument('--alg', type=str, default="L0",
                    help="gact optimization level, default does not quantize")
parser.add_argument('--input-size', type=int)
parser.add_argument('--get_macs', action='store_true')
parser.add_argument('--get_mem', action='store_true')
parser.add_argument('--get_speed', action='store_true')

best_acc1 = 0


def main():
    args = parser.parse_args()
    
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('scaled_wide_resnet'):
            model = scaled_wide_resnet(args.arch)
        elif args.arch.startswith('scaled_resnet'):
            model = scaled_resnet(args.arch)
        else:
            if args.arch in ['inception_v3']:
                kwargs = {"aux_logits": False}
            else:
                kwargs = {}
            model = models.__dict__[args.arch](**kwargs)

    model.cuda()

    if args.get_mem:
        print("========== Model Only ===========")
        usage = get_memory_usage(True)
        exp_recorder.record("network", args.arch)
        exp_recorder.record("algorithm", args.alg)
        exp_recorder.record("model_only", usage / GB, 2)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.input_size is None:
        if args.arch in ['inception_v3']:
            input_size = 299
        else:
            input_size = 224
    else:
        input_size = args.input_size

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if args.get_macs:
        from thop import profile
        input = torch.randn(1, 3, input_size, input_size).cuda()
        macs, params = profile(model, inputs=(input, ), custom_ops={})
        print(f"Macs: {macs}\t Params: {params}")
        out_file = "get_macs.json"
        with open(out_file, 'w') as fout:
            fout.write(json.dumps([macs, params]))
        print(f"save results to {out_file}")
        exit()

    train_model = model
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, train_model, criterion,
              optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


train_step_ct = 0
train_max_batch = 0
train_ips_list = []


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    ips = AverageMeter('IPS', ':.1f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ips],
        prefix="Epoch: [{}]".format(epoch))

    if args.benchmark == "exact":
        pass
    elif args.benchmark == "gact":
        gact.set_optimization_level(args.alg)
        controller = gact.controller.Controller(model)
        controller.install_hook()

    # switch to train mode
    model.train()
    for (i, (images, target)) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=False)
        target = target.cuda(args.gpu, non_blocking=False)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        
        if args.get_mem:
            print("========== Init Data Loader ===========")
            init_mem = get_memory_usage(True)
            exp_recorder.record("data_loader", init_mem /
                                GB - exp_recorder.val_dict['model_only'], 2)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if args.get_mem:
            torch.cuda.reset_peak_memory_stats()

            print("========== Before Backward ===========")
            before_backward = get_memory_usage(True)
            act_mem = get_memory_usage() - init_mem - \
                compute_tensor_bytes([loss, output])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.benchmark == "gact": 
                controller.iterate(None)
            del loss

            print("========== After Backward ===========")
            after_backward = get_memory_usage(True)
            total_mem = before_backward + (after_backward - init_mem)
            ref_act = before_backward - after_backward
            peak_mem = torch.cuda.max_memory_allocated()
            res = "Batch size: %d\tTotal Mem: %.2f MB\tAct Mem: %.2f MB\tRef Mem: %.2f\tPeak Mem: %.2f" % (
                len(output), total_mem / MB, act_mem / MB, ref_act / MB, peak_mem / MB)
            print(res)
            exp_recorder.record("batch_size", len(output))
            exp_recorder.record("total", total_mem / GB, 2)
            exp_recorder.record("activation", act_mem / GB, 2)
            exp_recorder.dump('mem_results.json')
            exit(0)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            optimizer.step()
        
        if i % args.print_freq == 0:
            progress.display(i)
        
        # measure elapsed time
        end.record()
        torch.cuda.synchronize()
        cur_batch_time = start.elapsed_time(end) / 1000.0 # event in ms

        # only use 8 batch size to get sensitivity
        def backprop():
            model.train()
            partial_bz = 8
            partial_image = images[:partial_bz, :, :, :]
            partial_target = target[:partial_bz]
            output = model(partial_image)
            loss = criterion(output, partial_target)
            optimizer.zero_grad()
            loss.backward()
            del loss
            del output
            del partial_image
            del partial_target
        if args.benchmark == "gact":
            controller.iterate(backprop)
        bs = len(images)
        del images
        train_ips_list.append(bs / cur_batch_time)
           
        if args.get_speed:
            global train_step_ct, train_max_batch
            train_max_batch = max(train_max_batch, bs)
            if train_step_ct >= 6:
                train_ips = np.median(train_ips_list)
                res = "BatchSize: %d\tIPS: %.2f\t,Cost: %.2f ms" % (
                    bs, train_ips, cur_batch_time)
                print(res, flush=True)
                exp_recorder.record("network", args.arch)
                exp_recorder.record("algorithm", args.alg)
                exp_recorder.record("benchmark", args.benchmark)
                exp_recorder.record("batch_size", train_max_batch)
                exp_recorder.record("ips", train_ips, 2)
                exp_recorder.record("tstamp", time.time(), 2)
                exp_recorder.dump('speed_results.json')
                exit(0)
            train_step_ct += 1
        
    if args.benchmark == "gact":
        controller.uninstall_hook()
    


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # Run a matmul to initialize cublas first
    a = torch.ones((1, 1)).cuda()
    a = (a @ a).cpu()
    del a
    torch.cuda.empty_cache()

    main()

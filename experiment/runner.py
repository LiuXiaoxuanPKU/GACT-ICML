import time
import os
import json
import torch
import torch.nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from util import AverageMeter
from actnn import get_memory_usage, compute_tensor_bytes
import actnn
import matplotlib.pyplot as plt


class Result:
    def __init__(self, modelname, bs, actnn_config,
                 batch_time, data_time,
                 peak_mem, total_mem, activation_mem,
                 loss) -> None:
        self.modelname = modelname
        self.batch_size = bs
        self.actnn_config = actnn_config
        self.batch_time = batch_time
        self.data_time = data_time
        self.loss = loss
        self.peak_mem = peak_mem
        self.total_mem = total_mem
        self.activation_mem = activation_mem


class Runner:
    def __init__(self, ty, bs, actnn_config, train_info, data_dir) -> None:
        self.get_speed = ty == 'speed'
        self.get_mem = ty == 'mem'
        if self.get_speed:
            self.iter = 20
        if self.get_mem:
            self.iter = 5
        print("self batch size", bs)
        self.batch_size = bs

        self.model_name = train_info['model']
        self.model = self.get_model(self.model_name)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), train_info['lr'],
                                         momentum=train_info['momentum'],
                                         weight_decay=train_info['weight_decay'])
        traindir = os.path.join(data_dir, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)

        self.actnn_config = actnn_config
        if int(actnn_config['bit']) == -1:
            self.actnn = False
        else:
            self.actnn = True
            self.bit = int(actnn_config['bit'])
            self.controller = actnn.controller.Controller(
                actnn_config['bit'], actnn_config['swap'], actnn_config['prefetch'])
            self.controller.filter_tensors(self.model.named_parameters())

    def get_model(self, model_name):
        return models.__dict__[model_name]().cuda()

    def pack_hook(self, x):
        if self.actnn:
            return self.controller.quantize(x)
        return x

    def unpack_hook(self, x):
        if self.actnn:
            return self.controller.dequantize(x)
        return x

    def run(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
            return self.run_helper()

    def run_helper(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        total_mem = AverageMeter('Total Memory', ':.4e')
        peak_mem = AverageMeter('Peak Memory', ':.4e')
        activation_mem = AverageMeter('Activation Memory', ':.4e')

        # train
        self.model.train()
        end = time.time()

        for i, (images, target) in enumerate(self.train_loader):
            images = images.cuda()
            target = target.cuda()

            print("iter: %d/%d" % (i, self.iter))
            # measure data loading time
            data_time.update(time.time() - end)
            if self.get_mem:
                print("===============After Data Loading=======================")
                init_mem = get_memory_usage(True)
                torch.cuda.reset_peak_memory_stats()

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item())

            if self.get_mem:
                print("===============Before Backward=======================")
                before_backward = get_memory_usage(True)
                activation_mem.update(
                    get_memory_usage() - init_mem - compute_tensor_bytes([loss, output]))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            del loss

            if self.get_mem:
                print("===============After Backward=======================")
                after_backward = get_memory_usage(True)
                total_mem.update(before_backward + (after_backward - init_mem))
                peak_mem.update(torch.cuda.max_memory_allocated())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.actnn:
                self.controller.iterate()

            if i == self.iter:
                break

        print(batch_time.summary())
        print(data_time.summary())
        print(losses.summary())
        print(peak_mem.summary())
        print(total_mem.summary())
        print(activation_mem.summary())
        return Result(self.model_name,
                      self.batch_size,
                      self.actnn_config,
                      batch_time, data_time,
                      peak_mem, total_mem, activation_mem,
                      losses)


class ResultProcessor:
    @staticmethod
    def cal_speed(bs, t):
        return bs/t

    @staticmethod
    def plot_speed(ax, rs):
        speeds = []
        batch_sizes = []
        label = None
        for r in rs:
            batch_sizes.append(r.batch_size)
            speeds.append(ResultProcessor.cal_speed(r.batch_size,
                                                    r.batch_time.get_value()))
            if int(r.actnn_config['bit']) != -1:
                label = "actnn %d bit" % int(r.actnn_config['bit'])
            else:
                label = "org"
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speed (IPS)")
        ax.plot(batch_sizes, speeds, label=label, marker='o', markersize=12)

    @staticmethod
    def plot_mem(ax, rs):
        mems = []
        batch_sizes = []
        label = None
        for r in rs:
            batch_sizes.append(r.batch_size)
            mems.append(r.activation_mem.get_value() / 1000000)
            if int(r.actnn_config['bit']) != -1:
                label = "actnn %d bit" % int(r.actnn_config['bit'])
            else:
                label = "org"
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Memory (MB)")
        ax.plot(batch_sizes, mems, label=label, marker='o', markersize=12)


class Engine:
    def __init__(self, config_filename) -> None:
        self.config = json.load(open(config_filename,  'r'))
        self.runners = []
        for actnn_config in self.config['actnn']:
            cur_runners = []
            for bs in self.config['batch_size']:
                runner = Runner(self.config['type'],
                                bs,
                                actnn_config,
                                self.config['train_info'],
                                self.config['data_dir'])
                cur_runners.append(runner)
            self.runners.append(cur_runners)
        self.results = []

    def start(self):
        for cur_runners in self.runners:
            cur_results = []
            for runner in cur_runners:
                cur_results.append(runner.run())
            self.results.append(cur_results)

    def plot(self, out):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        for cur_results in self.results:
            if self.config['type'] == 'speed':
                ResultProcessor.plot_speed(ax, cur_results)
            elif self.config['type'] == 'mem':
                ResultProcessor.plot_mem(ax, cur_results)
            else:
                print("[Error] Unsupport experiment type %s" %
                      self.config['type'])
        plt.legend()
        fig.savefig(out)
        plt.close(fig)

    def print(self):
        pass


if __name__ == "__main__":
    config = "./configs/resnet.json"
    engine = Engine(config)
    engine.start()
    engine.plot("mem.png")

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

MB = 1000000


class Result:
    def __init__(self, modelname, bs, actnn_config,
                 speed, data_time,
                 peak_mem, total_mem, activation_mem,
                 loss) -> None:
        self.modelname = modelname
        self.batch_size = bs
        self.actnn_config = actnn_config
        self.speed = speed
        self.data_time = data_time
        self.loss = loss
        self.peak_mem = peak_mem
        self.total_mem = total_mem
        self.activation_mem = activation_mem

    @staticmethod
    def from_json(obj):
        return Result(obj['model'], obj['batch_size'], obj['actnn_config'],
                      obj['speed'], obj['data_time'], obj['peak_mem'],
                      obj['total_mem'], obj['activation_mem'], obj['loss'])

    def dump(self):
        ret = {
            "model": self.modelname,
            "batch_size": self.batch_size,
            "actnn_config": self.actnn_config,
            "speed": self.speed,
            "data_time": self.data_time,
            "peak_mem": self.peak_mem,
            "total_mem": self.total_mem,
            "activation_mem": self.activation_mem,
            "loss": self.loss,
        }
        print(ret)
        return ret


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
                init_mem = get_memory_usage(True)  # model size + data size
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
            del output

            if self.get_mem:
                print("===============After Backward=======================")
                del images
                del target
                after_backward = get_memory_usage(True)  # model size
                # before backward : model size + data size + activation + loss + output
                # after backward : model size
                # init : model size + data size
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
                      self.batch_size / batch_time.get_value(), data_time.get_value(),
                      peak_mem.get_value() / MB, total_mem.get_value() / MB,
                      activation_mem.get_value() / MB, losses.get_value())


class ResultProcessor:
    @staticmethod
    def plot_line(ax, rs, field, color):
        values = []
        batch_sizes = []
        for r in rs:
            r = Result.from_json(r)
            batch_sizes.append(r.batch_size)
            values.append(r.__dict__[field])
        if "mem" in field:
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Memory (MB)")
        else:
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speed (IPS)")

        shapes = {"activation_mem": 'o', "total_mem": "^",
                  "peak_mem": "*", "speed": "+"}
        ax.plot(batch_sizes, values,
                label=r.actnn_config["label"] + " " + field,
                c=color, marker=shapes[field], markersize=12)


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

    def dump(self, out):
        total_results = []
        for cur_results in self.results:
            cur_results_dump = []
            for r in cur_results:
                cur_results_dump.append(r.dump())
            total_results.append(cur_results_dump)
        json.dump(total_results, open(
            out, 'w'), indent=4, ensure_ascii=True)

    def start(self):
        for cur_runners in self.runners:
            cur_results = []
            for runner in cur_runners:
                cur_results.append(runner.run())
            self.results.append(cur_results)

    def plot(self, infile, outfile, plot_field):
        data = json.load(open(infile))
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        colors = ['r', 'g', 'b', 'purple']
        for i, cur_results in enumerate(data):
            color = colors[i]
            for field in plot_field:
                ResultProcessor.plot_line(ax, cur_results, field, color)
        plt.legend()
        fig.savefig(outfile)
        plt.close(fig)


if __name__ == "__main__":
    # 2 4 8 speed
    config = "./configs/resnet_speed.json"
    engine = Engine(config)
    engine.start()
    output_data = "./results/resnet_speed"
    output_graph = "./results/resnet_speed_graph"
    engine.dump(output_data)
    engine.plot(output_data, output_graph, ["speed"])

    # 2 4 8 memory
    config = "./configs/resnet_mem.json"
    engine = Engine(config)
    engine.start()
    output_data = "./results/resnet_mem"
    output_graph = "./results/resnet_mem_graph"
    engine.dump(output_data)
    engine.plot(output_data, output_graph, ["activation_mem"])

    # swap + prefecth speed
    config = "./configs/resnet_swap_prefetch_speed.json"
    engine = Engine(config)
    engine.start()
    output_data = "./results/resnet_swap_prefetch_speed"
    output_graph = "./results/resnet_swap_prefetch_speed_graph_detail"
    engine.dump(output_data)
    engine.plot(output_data, output_graph, ["speed"])

    # swap +  prefetch memory
    config = "./configs/resnet_swap_prefetch_mem.json"
    engine = Engine(config)
    engine.start()
    output_data = "./results/resnet_swap_prefetch_mem"
    output_graph = "./results/resnet_swap_prefetch_mem_graph_detail"
    engine.dump(output_data)
    engine.plot(output_data, output_graph, [
                "activation_mem", "peak_mem", "total_mem"])

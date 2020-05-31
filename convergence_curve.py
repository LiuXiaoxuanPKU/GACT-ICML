import matplotlib
import json
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os


def ma(y):
    win = []
    result = []
    for i in range(len(y)):
        win.append(y[i])
        if len(win) > 7:
            win.pop(0)
        result.append(np.mean(win))

    return np.array(result)


def plot_curve(file_name, ax0, ax1, style, lab, max_ep):
    print(file_name)
    if not os.path.exists(file_name):
        return

    with open(file_name) as f:
        data = json.load(f)

    data = data['epoch']
    ep = data['ep']
    train_loss = data['train.loss']
    val_top1 = data['val.top1']
    ep = np.array(ep[:len(val_top1)])
    ep += max_ep - 1 - ep[-1]
    ax0.plot(ep, ma(train_loss), style, alpha=0.5, label=lab)
    ax1.plot(ep, ma(val_top1), style, alpha=0.5, label=lab)


styles = {8: 'r', 7: 'g--', 6: 'b-.', 5: 'c:', 4: 'm'}


def plot_setting(name, prefix, suffix, max_ep, ylim, qid, qlabel):
    xlim = [0, max_ep]

    fig0, ax0 = plt.subplots(1, figsize=(3.3, 3.3))
    fig1, ax1 = plt.subplots(1, figsize=(3.3, 3.3))

    plot_curve(prefix + 'a8w8g32{}/raport.json'.format(suffix), ax0, ax1, 'k:', 'QAT', 200)
    for i in range(8, 3, -1):
        plot_curve(prefix + 'a8w8g{}{}{}/raport.json'.format(i, qid, suffix),
                   ax0, ax1, styles[i], '{}-bit {}'.format(i, qlabel), max_ep)
    ax0.set_xlim(xlim)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax0.set_yscale('log')
    ax0.legend()
    ax1.legend()
    ax0.set_xlabel('Epochs')
    ax1.set_xlabel('Epochs')
    ax0.set_ylabel('Training Loss')
    ax1.set_ylabel('Validation Accuracy')
    l, b, w, h = ax0.get_position().bounds
    ax0.set_position([l + 0.1 * w, b + 0.1 * h, 0.9 * w, 0.9 * h])
    l, b, w, h = ax1.get_position().bounds
    ax1.set_position([l + 0.1 * w, b + 0.1 * h, 0.9 * w, 0.9 * h])
    fig0.savefig(name + '_loss.pdf', transparent=True)
    fig1.savefig(name + '_acc.pdf', transparent=True)


plot_setting('cifar10_ptq', '/home/jianfei/work/RN50v1.5/reports/cifar10/', '_seed0', 200, [80, 95], '', 'PTQ')
plot_setting('cifar10_psq', '/home/jianfei/work/RN50v1.5/reports/cifar10/', '_seed0', 200, [80, 95], '_p', 'PSQ')
plot_setting('cifar10_bhq', '/home/jianfei/work/RN50v1.5/reports/cifar10/', '_seed0', 200, [80, 95], '_h', 'BHQ')
plot_setting('resnet18_ptq', '/home/jianfei/work/RN50v1.5/reports/resnet18/resnet18_', '', 90, [55, 75], '', 'PTQ')
plot_setting('resnet18_psq', '/home/jianfei/work/RN50v1.5/reports/resnet18/resnet18_', '', 90, [55, 75], '_p', 'PSQ')
plot_setting('resnet18_bhq', '/home/jianfei/work/RN50v1.5/reports/resnet18/resnet18_', '', 90, [55, 75], '_h', 'BHQ')
plot_setting('resnet50_ptq', '/home/jianfei/work/RN50v1.5/reports/resnet50/resnet50_', '', 90, [60, 80], '', 'PTQ')
plot_setting('resnet50_psq', '/home/jianfei/work/RN50v1.5/reports/resnet50/resnet50_', '', 90, [60, 80], '_p', 'PSQ')
plot_setting('resnet50_bhq', '/home/jianfei/work/RN50v1.5/reports/resnet50/resnet50_', '', 90, [60, 80], '_h', 'BHQ')
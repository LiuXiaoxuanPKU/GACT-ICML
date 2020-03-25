import matplotlib
import json
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def ma(y):
    win = []
    result = []
    for i in range(len(y)):
        win.append(y[i])
        if len(win) > 7:
            win.pop(0)
        result.append(np.mean(win))

    return np.array(result)

def plot_curve(file_name, ax0, ax1, style, lab):
    with open(file_name) as f:
        data = json.load(f)

    data = data['epoch']
    ep = data['ep']
    train_loss = data['train.loss']
    val_top1 = data['val.top1']
    ep = ep[:len(val_top1)]
    ax0.plot(ep, ma(train_loss), style, alpha=0.5, label=lab)
    ax1.plot(ep, ma(val_top1), style, alpha=0.5, label=lab)


fig, ax = plt.subplots(2, figsize=(10, 20))
plot_curve('results/preact_resnet56_0/raport.json', ax[0], ax[1], 'k-', 'exact')
plot_curve('results/preact_resnet56_f32b6bw8_hadamard/raport.json', ax[0], ax[1], 'r-', '32_6_H')
plot_curve('results/preact_resnet56_f32b5bw8_hadamard/raport.json', ax[0], ax[1], 'g-', '32_5_H')
plot_curve('results/preact_resnet56_f32b4bw8_hadamard/raport.json', ax[0], ax[1], 'b-', '32_4_H')
plot_curve('results/preact_resnet56_f32b6bw8_persample/raport.json', ax[0], ax[1], 'r--', '32_6_P')
plot_curve('results/preact_resnet56_f32b5bw8_persample/raport.json', ax[0], ax[1], 'g--', '32_5_P')
plot_curve('results/preact_resnet56_f32b4bw8_persample/raport.json', ax[0], ax[1], 'b--', '32_4_P')
plot_curve('results/preact_resnet56_f32b7bw8/raport.json', ax[0], ax[1], 'k.', '32_7')
plot_curve('results/preact_resnet56_f32b6bw8/raport.json', ax[0], ax[1], 'r.', '32_6')
plot_curve('results/preact_resnet56_f32b5bw8/raport.json', ax[0], ax[1], 'g.', '32_5')
plot_curve('results/preact_resnet56_f6b6bw8_hadamard/raport.json', ax[0], ax[1], 'r:', '6_6_H')
plot_curve('results/preact_resnet56_f5b5bw8_hadamard/raport.json', ax[0], ax[1], 'g:', '5_5_H')
ax[0].set_yscale('log')

ax[0].legend()
ax[1].legend()
ax[0].set_ylabel('train loss')
ax[1].set_ylabel('val top1')
ax[1].set_ylim([90.0, 94.0])
fig.savefig('convergence.pdf')

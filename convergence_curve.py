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
    if lab.find('4x') != -1:
        ep = np.array(ep) / 4
    ax0.plot(ep, ma(train_loss), style, alpha=0.5, label=lab)
    ax1.plot(ep, ma(val_top1), style, alpha=0.5, label=lab)


fig, ax = plt.subplots(2, figsize=(10, 20))
# plot_curve('results/20200325_9a367/32/raport.json', ax[0], ax[1], 'k:', 'exact')
# plot_curve('results/20200325_9a367/a6w6g4_h/raport.json', ax[0], ax[1], 'r-', 'a6w6g4')
# plot_curve('results/20200325_9a367/a5w5g4_h/raport.json', ax[0], ax[1], 'g-', 'a5w5g4')
# plot_curve('results/20200325_9a367/a4w4g4_h/raport.json', ax[0], ax[1], 'b-', 'a4w4g4')
plot_curve('results/20200325_9a367/a8w4g4_h/raport.json', ax[0], ax[1], 'c:', '4 bits Householder')
# plot_curve('results/20200325_9a367/a6w6/raport.json', ax[0], ax[1], 'r:', 'a6w6')
# plot_curve('results/20200325_9a367/a5w5/raport.json', ax[0], ax[1], 'g:', 'a5w5')
# plot_curve('results/20200325_9a367/a4w4/raport.json', ax[0], ax[1], 'b:', 'a4w4')
plot_curve('results/20200325_9a367/a8w4/raport.json', ax[0], ax[1], 'k-', 'exact')
# plot_curve('results/20200325_9a367/a8w4g9/raport.json', ax[0], ax[1], 'm-', 'a8w4g9')
plot_curve('results/20200325_9a367/a8w4g8/raport.json', ax[0], ax[1], 'y-', '8 bits')
plot_curve('results/20200325_9a367/a8w4g7/raport.json', ax[0], ax[1], 'r-', '7 bits')
# plot_curve('results/20200325_9a367/a8w4g7_4x/raport.json', ax[0], ax[1], 'r--', 'a8w4g7_4x')
plot_curve('results/20200325_9a367/a8w4g6/raport.json', ax[0], ax[1], 'g-', '6 bits')
# plot_curve('results/20200325_9a367/a8w4g6_4x/raport.json', ax[0], ax[1], 'g--', 'a8w4g6_4x')
plot_curve('results/20200325_9a367/a8w4g5/raport.json', ax[0], ax[1], 'b-', '5 bits')
# plot_curve('results/20200325_9a367/a8w4g5_4x/raport.json', ax[0], ax[1], 'b--', 'a8w4g5_4x')
ax[0].set_yscale('log')

ax[0].legend()
ax[1].legend()
ax[0].set_ylabel('train loss')
ax[1].set_ylabel('val top1')
ax[1].set_ylim([90.0, 94.0])
fig.savefig('convergence.pdf')

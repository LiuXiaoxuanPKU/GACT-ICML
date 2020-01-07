import matplotlib
import json
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_curve(file_name, ax0, ax1, lab):
    with open(file_name) as f:
        data = json.load(f)

    data = data['epoch']
    ep = data['ep']
    train_loss = data['train.loss']
    val_top1 = data['val.top1']
    ep = ep[:len(val_top1)]
    ax0.plot(ep, train_loss, alpha=0.5, label=lab)
    ax1.plot(ep, val_top1, alpha=0.5, label=lab)


fig, ax = plt.subplots(2, figsize=(5, 10))
plot_curve('results/preact_resnet56/raport.json', ax[0], ax[1], 'exact')
# plot_curve('results/preact_resnet56_8bit_persample/raport.json', ax[0], ax[1], '8bit')
# plot_curve('results/preact_resnet56_6bit_persample/raport.json', ax[0], ax[1], '6bit')
plot_curve('results/preact_resnet56_4bit_persample/raport.json', ax[0], ax[1], '4bit')
plot_curve('results/preact_resnet56_5bits_ortho_chw/raport.json', ax[0], ax[1], '5bit_chw')
plot_curve('results/preact_resnet56_4bits_ortho_chw/raport.json', ax[0], ax[1], '4bit_chw')
# plot_curve('results/preact_resnet56_8bit_persample_biased/raport.json', ax[0], ax[1], '8bit_biased')
ax[0].set_yscale('log')

ax[0].legend()
ax[1].legend()
ax[0].set_ylabel('train loss')
ax[1].set_ylabel('val top1')
ax[1].set_ylim([85.0, 94.0])
fig.savefig('convergence.pdf')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

num_seeds = 3
configs = ['4bit', #'exact', '4bit', '2bit',
           # '2bit_psr', '2bit_ps', '2bit_pl',
           '2bit_psr_nobn', '2bit_ps_nobn', '2bit_pl_nobn']
data = []
for config in configs:
    for seed in range(num_seeds):
        with open('cifar_train_large/{}_seed{}.log'.format(config, seed)) as f:
            epoch = 0
            window = []
            for line in f:
                if line.find('val.top1') != -1:
                    line = line.split()
                    top1 = float(line[-7])
                    loss = float(line[11])
                    window.append(top1)
                    if len(window) > 5:
                        window.pop(0)
                    data.append((epoch, top1, np.mean(window), loss, config, seed))
                    epoch += 1

data = pd.DataFrame(data, columns=['epoch', 'acc', 'avgacc', 'loss', 'config', 'seed'])
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
sns.lineplot(data=data, x='epoch', y='avgacc', hue='config', ax=ax[0])
ax[0].set_ylim([80, 94])
sns.lineplot(data=data, x='epoch', y='loss', hue='config', ax=ax[1])
ax[1].set_yscale('log')
fig.savefig('convergence.pdf')

import numpy as np
import matplotlib
import json
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

folders = ['./', 'g15', 'g16']

# styles = ['c.:', 'rx-', 'g^--', 'bv-']
# data = 'imagenet'

samplevar = 0

methods = []
abits = []
accs = []

for method in ['naive', 'pg', 'ps', 'pl']:
    if method == 'naive' or method == 'pg':
        bits = [2, 3, 4]
    else:
        bits = [1.5, 1.75, 2, 2.5, 3, 4]

    for bit in bits:
        for seed in range(5):
            for folder in folders:
                file_name = '{}/{}_{}_seed{}/raport.json'.format(
                    folder, method, bit, seed)

                try:
                    data = json.load(open(file_name))
                    val = data['epoch']['val.top1']
                    if len(val) == 200:
                        methods.append(method)
                        abits.append(bit)
                        accs.append(val[-1])
                        methods.append(method)
                        abits.append(bit)
                        accs.append(val[-2])
                except Exception:
                    pass

        print(method, bit, np.mean(accs), np.std(accs), len(accs))

fig, ax = plt.subplots(figsize=(5, 5))
sns.lineplot(x=abits, y=accs, hue=methods, ax=ax)
ax.legend()
# ax.set_yscale('log')
ax.set_ylim([90, 94])
fig.savefig('acc_cifar10.pdf')

        # file_name = 'grad_{}/{}_{}_seed0.log'.format(data, method, bit)
        # with open(file_name) as f:
        #     for line in f:
        #         if line.find('SampleVar') != -1:
        #             line = line.replace(',', '').split()
        #             gradvars.append(float(line[6]))
        #             if method == 'naive' and bit == 4:
        #                 samplevar = float(line[9])

#     ax.plot(bits, gradvars, style, label=method)
#
# ax.plot([1.5, 4], [samplevar, samplevar], 'k', label='sample')
#
# ax.legend()


import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# for epochs in [10, 70]:
#     file_name = '../resnet/resnet50_{}epochs.txt'.format(epochs)

fig, ax = plt.subplots(figsize=(5, 5))
styles = ['c.:', 'rx-', 'g^--', 'bv-']
data = 'imagenet'

samplevar = 0
for method, style in zip(['naive', 'pg', 'ps', 'pl'], styles):
    if method == 'naive' or method == 'pg':
        bits = [2, 3, 4]
    else:
        bits = [1.5, 1.75, 2, 2.5, 3, 4]

    gradvars = []

    for bit in bits:
        file_name = 'grad_{}/{}_{}_seed0.log'.format(data, method, bit)
        with open(file_name) as f:
            for line in f:
                if line.find('SampleVar') != -1:
                    line = line.replace(',', '').split()
                    gradvars.append(float(line[6]))
                    if method == 'naive' and bit == 4:
                        samplevar = float(line[9])

    ax.plot(bits, gradvars, style, label=method)

ax.plot([1.5, 4], [samplevar, samplevar], 'k', label='sample')

ax.legend()
ax.set_yscale('log')
fig.savefig('var_{}.pdf'.format(data))

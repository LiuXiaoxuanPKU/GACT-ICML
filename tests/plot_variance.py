import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

for samples in [10]:
    file_name = '../resnet/qall_variance_200epoch_{}samples.txt'.format(samples)
    with open(file_name) as f:
        grad_norm = []
        bias = []
        var = []
        error = []
        for line in f:
            if line.find('grad norm') != -1:
                line = line.replace(',', '').split()
                grad_norm.append(float(line[4]))
                bias.append(float(line[7]))
                var.append(float(line[10]))
                error.append(float(line[13]))

        fig, ax = plt.subplots()
        ax.plot(grad_norm, label='grad_norm')
        ax.plot(bias, label='bias')
        ax.plot(var, label='var')
        ax.plot(error, label='error')
        ax.legend()
        ax.set_yscale('log')
        fig.savefig('var{}.pdf'.format(samples))
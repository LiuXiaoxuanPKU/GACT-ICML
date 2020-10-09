import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# for epochs in [10, 70]:
#     file_name = '../resnet/resnet50_{}epochs.txt'.format(epochs)

for model in ['resnet56', 'preact_resnet56']:
    file_name = '../resnet/{}_190epochs.txt'.format(model)
    grad_norm = []
    bias = []
    var = []
    error = []
    with open(file_name) as f:
        for line in f:
            if line.find('grad norm') != -1:
                line = line.replace(',', '').split()
                grad_norm.append(float(line[4]))
                bias.append(float(line[7]))
                var.append(float(line[10]))
                error.append(float(line[13]))

    # file_name = '../resnet/resnet50_trace_{}.txt'.format(epochs)
    # trace = []
    # with open(file_name) as f:
    #     for line in f:
    #         if line.find('tensor') != -1:
    #             line = line.replace(',', ' ').replace('(', ' ').split()
    #             trace.append(np.abs(float(line[2])))

    # fig, axs = plt.subplots(2, figsize=(5, 10))
    # ax = axs[0]

    fig, ax = plt.subplots()
    ax.plot(grad_norm, label='grad_norm')
    ax.plot(bias, label='bias')
    ax.plot(var, label='var')
    ax.plot(error, label='error')
    ax.legend()
    ax.set_yscale('log')
    ax.set_title('grad norm')

    # ax = axs[1]
    # ax.plot(trace)
    # ax.set_yscale('log')
    # ax.set_title('Hessian trace')
    fig.savefig('var{}.pdf'.format(model))
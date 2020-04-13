import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

xs = range(4, 9)
for epoch in [0, 100, 200]:
    fig, ax = plt.subplots()
    for alg, prefix in [('naive', ''), ('persample', '_p'), ('householder', '_h')]:
        sample_vars = []
        quant_vars = []
        for x in xs:
            with open('g{}_e{}{}.log'.format(x, epoch, prefix)) as f:
                for line in f:
                    if line.find('Overall Var') != -1:
                        line = line.replace(',', '').split()
                        sample_vars.append(float(line[-1]))
                        quant_vars.append(float(line[-5]))

        if alg == 'naive':
            ax.plot(list(xs), sample_vars, label='sample')
        ax.plot(list(xs), quant_vars, label=alg)

    ax.legend()
    ax.set_yscale('log')
    ax.grid()
    ax.set_xticks(list(xs))
    fig.savefig('stds_{}.pdf'.format(epoch))

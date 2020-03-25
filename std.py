import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
xs = range(4, 9)
for alg, prefix in [('naive', '{}.log'), ('persample', '{}_persample.log'), ('householder', '{}_householder.log')]:
    sample_vars = []
    quant_vars = []
    for x in xs:
        with open(prefix.format(x)) as f:
            for line in f:
                if line.find('SampleVar') != -1:
                    line = line.replace(',', '').split()
                    sample_vars.append(float(line[2]))
                    quant_vars.append(float(line[5]))

    if alg == 'naive':
        ax.plot(list(xs), sample_vars, label='sample')
    ax.plot(list(xs), quant_vars, label=alg)

ax.legend()
ax.set_yscale('log')
ax.grid()
ax.set_xticks(list(xs))
fig.savefig('stds.pdf')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

xs = range(4, 9)
for epoch in [100]: #[0, 100, 200]:
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    for alg, prefix, style in [('PTQ', '', 'r^-'), ('PSQ', '_p', 'gx-'), ('BHQ', '_h', 'bv-')]:
        sample_vars = []
        quant_vars = []
        for x in xs:
            with open('g{}_e{}{}.log'.format(x, epoch, prefix)) as f:
                for line in f:
                    if line.find('OverallVar') != -1:
                        line = line.replace(',', '').split()
                        sample_vars.append(float(line[2]))
                        quant_vars.append(float(line[5]))

        if alg == 'PTQ':
            ax.plot(list(xs), [sample_vars[0]]*5, 'k-', label='QAT')
        ax.plot(list(xs), quant_vars, style, label=alg)
        print(sample_vars)
        print(quant_vars)

    ax.legend()
    ax.set_yscale('log')
    ax.grid()
    ax.set_xticks(list(xs))
    ax.set_xlim([8, 4])
    ax.set_xlabel('Bitwidth')
    ax.set_ylabel('Variance')
    l, b, w, h = ax.get_position().bounds
    ax.set_position([l + 0.07 * w, b + 0.05 * h, 0.97 * w, 0.95 * h])
    fig.savefig('stds_{}.pdf'.format(epoch), transparent=True)

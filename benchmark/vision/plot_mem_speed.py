import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})
import seaborn as sns

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

capuchin_resnet50 = {
  96: 0.99,
  128: 0.99,
  160: 0.90,
  192: 0.84,
  224: 0.77,
  256: 0.70, 
  288: 0.63,
}

#capuchin_resnet50 = {k+32: v for k, v in capuchin_resnet50.items()}

def read_data(in_file):
    ret = {}  # network -> algorithm -> batch_size -> ips

    for line in open(in_file):
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue

        val_dict = json.loads(line)

        network, alg, batch_size, ips = val_dict['network'], val_dict['algorithm'],\
                val_dict['batch_size'], val_dict['ips']

        if ips < 0:
            continue
        if batch_size >= 256 and batch_size % 32 != 0:
            continue

        if network not in ret:
            ret[network] = dict()
        if alg not in ret[network]:
            ret[network][alg] = dict()
        if batch_size not in ret[network][alg]:
            ret[network][alg][batch_size] = 0

        ret[network][alg][batch_size] = max(ret[network][alg][batch_size], ips)

    # # add capuchin
    # network = 'resnet50'
    # alg = 'capuchin'
    # ret[network][alg] = dict()
    # for batch_size in capuchin_resnet50:
    #     ret[network][alg][batch_size] = capuchin_resnet50[batch_size] * ret[network]['exact'][128]

    # # add quantize-best
    # best_name = 'quantize-best'
    # to_delete = []
    # for network in ret:
    #     ret[network][best_name] = dict()
    #     for alg in ret[network]:
    #         if not alg.startswith('quantize'):
    #             continue
    #         for batch_size in ret[network][alg]:
    #             ips = ret[network][best_name].get(batch_size, 0)
    #             ret[network][best_name][batch_size] = max(ret[network][alg][batch_size], ips)

    return ret

def show_name(name):
    replace_dict = {
        'swap': 'swap',
        'dtr': 'DTR',
        'blpa': 'BLPA',
        'capuchin': 'CAP',
        'monet': 'MONet',
        'exact': 'FP32',
        'quantize-best': 'ActNN',
    }

    return replace_dict.get(name, name)


def method_to_marker(method):
    return '.'

def method_to_color(method):
    val_dict = {
        'FP32':'C1',
        'None': 'C1',
        'L1': 'C2',
        'L1.2': 'C3',
        'dtr' : 'C4',
        'actnn-L1': 'C5',
        'actnn-L2': 'C6',
        'actnn-L3': 'C7',
        'actnn-L4': 'C8',
    }
    if method in val_dict:
        return val_dict[method]
    else:
        return 'C2'

order_list = ['exact', 'dtr', "blpa", "capuchin", "swap"]
def method_to_order(method):
    if method in order_list:
        return "_" + str(order_list.index(method))
    else:
        return method


def get_max_batch_size(data, alg, network):
    max_batch = 0
    max_ips = 0
    last_ips = 0
    for batch_size, ips in data[network][alg].items():
        if ips > 0:
            max_batch = max(max_batch, batch_size)
            max_ips = max(max_ips, ips)
            last_ips = ips
    return max_batch, max_ips, last_ips


def get_seg_points(data, network):
    methods = list(data[network].keys())
    quantize_methods = [m for m in methods if m.startswith('quantize') and m != 'quantize-best']

    batch_sizes = list(data[network]['quantize-best'].keys())
    batch_sizes.sort()

    pre_alg = None
    seg_points = []
    for batch_size in batch_sizes:
        for alg in quantize_methods:
            if batch_size not in data[network][alg]:
                continue
            if data[network][alg][batch_size] == data[network]['quantize-best'][batch_size]:
                break

        if alg != pre_alg:
            seg_points.append(batch_size)
            pre_alg = alg

    seg_points[0] = 0
    if network in ['resnet152', 'wide_resnet101_2']:
        del seg_points[4]
    seg_points.append(batch_size)
    return seg_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="png")
    args = parser.parse_args()

    tmp_file = "tmp.json"

    run_cmd(f"rm -rf {tmp_file}")
    for in_file in glob.glob("speed_mem/*.json"):
        run_cmd(f"cat {in_file} >> {tmp_file}")

    data = read_data(tmp_file)

    # for network in ['resnet50', 'resnet152', 'densenet201', 'wide_resnet101_2']:
    for network in ['resnet50', 'resnet152']:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        algs = list(data[network].keys())
        algs.sort(key=method_to_order)

        # draw curves
        for alg in algs:
            if alg == 'FP32' or alg == 'actnn-L3.1':
                continue
            # if alg.startswith('quantize') and alg != 'quantize-best':
            #     continue
            # if alg == 'exact':
            #     continue

            xs = list(data[network][alg].keys())
            xs.sort()
            ys = [data[network][alg][x] for x in xs]
            ax.plot(xs, ys, label=show_name(alg), color=method_to_color(alg),
                    marker=method_to_marker(alg), markersize=4)
            if len(xs) > 1:
                xs = [xs[-1], xs[-1] + (xs[-1] - xs[-2])]
            else:
                xs = [xs[-1], xs[-1] + (xs[-1] - 0)]
            ys = [ys[-1], ys[-1]]
            ax.plot(xs, ys, color='gray', linestyle=':')
            ax.plot(xs[-1], ys[-1], color='C3', marker='x', markersize=6, zorder=100)

        # draw full precision region
        exact_max_batch, exact_max_ips, _ = get_max_batch_size(data, "None", network)
        max_batch, max_ips, last_ips = get_max_batch_size(data, 'L1.2', network)
        _, _, swap_last_ips = get_max_batch_size(data, 'dtr', network)
        # _, _, swap_last_ips = get_max_batch_size(data, 'rotor', network)
        
        y_max = max(max_ips, exact_max_ips) * 1.2
        ax.bar([exact_max_batch / 2], [y_max], width=exact_max_batch, alpha=0.2,
                color=method_to_color('exact'))
        ax.set_ylim(bottom=0, top=y_max)
        ax.set_xlim(left=0)

        # draw ratio
        y_val = swap_last_ips // 2
        ax.arrow(x=exact_max_batch, y=y_val, dx=max_batch - exact_max_batch - 10, dy=0,
                 width=0.01, head_width=y_max / 40, head_length=max_batch / 60, fc='black', ec='black')
        ax.annotate('$%.1f \\times$' % (max_batch / exact_max_batch), xy=(max_batch * 2 / 3, y_val + 2))
        ax.arrow(x=max_batch, y=y_val, dx=0, dy=last_ips-y_val, width=0.01, linestyle=(0, (1, y_val)),
                 color='gray')

        # draw grid
        ax.xaxis.grid(linewidth=0.4, linestyle='dotted')

        # # draw opt level
        # seg_points = get_seg_points(data, network)
        # for i in range(len(seg_points) - 1):
        #     x = (seg_points[i] + seg_points[i+1]) / 2
        #     y = data[network]['quantize-best'][x // 32 * 32]
        #     ax.annotate('L%d' % i, xy=(x, y + y_max/13), ha='center', va='center')

        # draw legend
        ax.legend(loc="upper left", ncol=1, columnspacing=0.8, handlelength=1.0,
                  handletextpad=0.4, bbox_to_anchor=(1.0, 1.0), fontsize=9)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Training Throughput')

        out_file = f"mem_speed_{network}.{args.suffix}"
        fig.savefig(out_file, bbox_inches='tight')
        print(f"Output the plot to {out_file}")

    run_cmd(f"rm -rf {tmp_file}")

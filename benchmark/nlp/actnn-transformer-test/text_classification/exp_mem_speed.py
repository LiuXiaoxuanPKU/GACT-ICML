import argparse
import json
import os
import time


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)




# bert-large-cased
def network_to_command(network):
    cmd = "python run_glue.py --model_name_or_path ARCH --task_name sst2 --max_length 128 " + \
        "--per_device_train_batch_size BS --per_device_eval_batch_size 128 --learning_rate 1e-5 " + \
        "--num_train_epochs 1 --seed 42 --pad_to_max_length "
    cmd = cmd.replace("ARCH", network)
    return cmd

def run_benchmark(network, alg, batch_size, debug_mem=False, debug_speed=False, input_size=None, layer_num=None):
    os.environ['DEBUG_MEM'] = str(debug_mem)
    os.environ['DEBUG_SPEED'] = str(debug_speed)
    cmd = network_to_command(network)
    cmd = cmd.replace("BS", f"{batch_size}")
    
    if alg != None:
        cmd += " --output_dir log/sst2/LEVEL/ --actnn --opt_level LEVEL ".replace("LEVEL", alg)
        
    if debug_speed:
        cmd += " --get_speed "
    
    if layer_num is not None:
        cmd += f" --layer_num {layer_num}"

    ret_code = run_cmd(cmd)

    if ret_code != 0:
        out_file = "speed_results.json"
        with open(out_file, "a") as fout:
            val_dict = {
                "network": network,
                "algorithm": alg,
                "batch_size": batch_size,
                "ips": -1,
            }
            fout.write(json.dumps(val_dict) + "\n")
        print(f"save results to {out_file}")

    time.sleep(1)
    run_cmd("nvidia-smi > /dev/null")
    time.sleep(1)
    return ret_code


def round_up(x):
    return int((x + 3) // 4 * 4)


def round_down(x):
    return int(x // 4 * 4)


def binary_search_max_batch(network, alg, low, high):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        success = run_benchmark(network, alg, mid, debug_speed=True) == 0
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def binary_search_max_input_size(alg, low, high, network, batch_size):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        success = (run_benchmark(network, alg, input_size=mid, batch_size=batch_size,
                                 debug_speed=True) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def binary_search_max_layer(alg, low, high, batch_size):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        network = "scaled_resnet_%d" % mid
        success = (run_benchmark(
            network, alg, batch_size=batch_size, debug_speed=True, layer_num=mid) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def binary_search_max_width(alg, low, high, batch_size):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        network = "scaled_wide_resnet_%d" % mid
        success = (run_benchmark(
            network, alg, batch_size=batch_size, debug_speed=True) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def get_ips(network, alg, batch_size, input_size=None):
    run_benchmark(network, alg, batch_size,
                  input_size=input_size, debug_speed=True)
    line = list(open("speed_results.json").readlines())[-1]
    return json.loads(line)['ips']


def get_macs(network, alg, batch_size, input_size=None):
    run_benchmark(network, alg, batch_size,
                  input_size=input_size, get_macs=True)
    line = list(open("get_macs.json").readlines())[-1]
    return json.loads(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='linear_scan')
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'linear_scan':
        networks = ['bert-large-cased']
        # batch_sizes = list(range(4, 20, 4)) + list(range(20, 240, 8))
        batch_sizes = list(range(236, 300, 16))

        algs = ['L1.2']
    else:
        networks = ['bert-large-cased']
        algs = ['L1', 'L1.2']

    if args.mode == 'linear_scan':
        for network in networks:
            for alg in algs:
                failed = 0
                for batch_size in batch_sizes:
                    if run_benchmark(network, alg, batch_size, debug_mem=False, debug_speed=True) != 0:
                        if failed >= args.retry:
                            break
                        failed += 1
                                               
    elif args.mode == 'binary_search_max_batch':
        for network in networks:
            for alg in algs:
                low, high = 16, 1024
                max_batch_size = binary_search_max_batch(
                    network, alg, low, high)
                ips = get_ips(network, alg, max_batch_size)

                out_file = "max_batch_results.json"
                with open(out_file, "a") as fout:
                    val_dict = {
                        "network": network,
                        "algorithm": alg,
                        "max_batch_size": max_batch_size,
                        "ips": ips,
                        "tstamp": time.time()
                    }
                    fout.write(json.dumps(val_dict) + "\n")
                print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_input_size':
        for alg in algs:
            low, high = 224, 768
            batch_size = 64
            network = 'resnet152'
            max_input_size = binary_search_max_input_size(
                alg, low, high, network, batch_size)
            ips = get_ips(network, alg, batch_size, input_size=max_input_size)
            # macs, params = get_macs(
            #     network, alg, batch_size, input_size=max_input_size)

            out_file = "max_input_size_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_input_size": max_input_size,
                    "ips": ips,
                    # "macs": macs,
                    # "params": params,
                    # "TFLOPS": round(macs * ips / 1e12, 2),
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_layer':
        for alg in algs:
            low, high = 24, 256
            batch_size = 64
            max_layer = binary_search_max_layer(alg, low, high, batch_size)
            network = 'scaled_bert_%d' % max_layer
            ips = get_ips(network, alg, batch_size)
            macs, params = get_macs(network, alg, batch_size)

            out_file = "max_layer_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_layer": max_layer,
                    "ips": ips,
                    "macs": macs,
                    "params": params,
                    "TFLOPS": round(macs * ips / 1e12, 2),
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_width':
        for alg in algs:
            low, high = 64, 512
            batch_size = 64
            max_width = binary_search_max_width(
                alg, low, high, batch_size=batch_size)
            network = 'scaled_wide_resnet_%d' % max_width
            ips = get_ips(network, alg, batch_size)
            macs, params = get_macs(network, alg, batch_size)

            out_file = "max_width_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_width": max_width,
                    "ips": ips,
                    "macs": macs,
                    "params": params,
                    "TFLOPS": round(macs * ips / 1e12, 2),
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")

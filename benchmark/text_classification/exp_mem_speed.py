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

def run_benchmark(network, alg, batch_size, debug_mem=False, debug_speed=False,
                hidden_size=None, layer_num=None, intermediate_size=None, get_macs=False):
    os.environ['DEBUG_SPEED'] = str(debug_speed)
    cmd = network_to_command(network)
    cmd = cmd.replace("BS", f"{batch_size}")
    
    if alg == 'ckpt':
        cmd += " --ckpt "
    elif alg == 'L1_ckpt':
        cmd += " --ckpt "
        cmd += " --gact --opt_level L1 "
    elif alg == "L1.4_ckpt":
        cmd += " --ckpt "
        cmd += " --gact --opt_level L1.4 "
    elif alg != None:
        cmd += " --output_dir log/sst2/LEVEL/ --gact --opt_level LEVEL ".replace("LEVEL", alg)
        
    if debug_speed:
        cmd += " --get-speed "
    
    if debug_mem:
        cmd += " --get-mem "
    
    if intermediate_size is not None:
        cmd += f" --customize "
        cmd += f" --intermediate_size {intermediate_size}"

    if layer_num is not None:
        cmd += f" --customize "
        cmd += f" --layer_num {layer_num}"
    
    if hidden_size is not None:
        cmd += f" --customize "
        cmd += f" --hidden_size {hidden_size}"

    if get_macs:
        cmd += " --get_macs "

    ret_code = run_cmd(cmd)

    if ret_code != 0:
        out_file = "speed_results.json"
        with open(out_file, "a") as fout:
            val_dict = {
                "network": network,
                "algorithm": alg,
                "batch_size": batch_size,
                "layer_num": layer_num,
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

def round_up_16(x):
    return int((x + 15) // 16 * 16)

def round_down(x):
    return int(x // 4 * 4)

def round_down_16(x):
    return int(x // 16 * 16)

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


def binary_search_max_hidden_size(alg, low, high, network, batch_size):
    ret = 0
    low, high = round_up_16(low), round_down_16(high)

    while low <= high:
        mid = round_down_16(low + (high - low) // 2)
        success = (run_benchmark(network, alg, hidden_size=mid, batch_size=batch_size,
                                 debug_speed=True) == 0)
        if success:
            ret = mid
            low = round_up_16(mid + 1)
        else:
            high = round_down_16(mid - 1)

    return ret


def binary_search_max_layer(alg, low, high, batch_size):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        network = "bert-large-cased"
        success = (run_benchmark(
            network, alg, batch_size=batch_size, debug_speed=True, layer_num=mid) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def binary_search_max_intermediate_size(alg, low, high, batch_size):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        network = "bert-large-cased"
        success = (run_benchmark(
            network, alg, batch_size=batch_size, debug_speed=True, intermediate_size=mid) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def get_ips(network, alg, batch_size, hidden_size=None, layer_num=None, intermediate_size=None):
    run_benchmark(network, alg, batch_size, layer_num=layer_num,
                  hidden_size=hidden_size, debug_speed=True, intermediate_size=intermediate_size)
    line = list(open("speed_results.json").readlines())[-1]
    return json.loads(line)['ips']


def get_macs(network, alg, batch_size, hidden_size=None, layer_num=None, intermediate_size=None):
    run_benchmark(network, alg, batch_size, layer_num=layer_num,
                  hidden_size=hidden_size, get_macs=True, intermediate_size=intermediate_size)
    line = list(open("get_macs.json").readlines())[-1]
    return json.loads(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='linear_scan')
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'linear_scan':
        networks = ['bert-base-cased']
        batch_sizes = list(range(4, 20, 4)) + list(range(20, 240, 8))
        algs = ["L1", "L2", "L1_ckpt"]
    else:
        networks = ['bert-large-cased']
        algs = [None, 'L1', 'L2']

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
    elif args.mode == 'binary_search_max_hidden_size':
        for alg in algs:
            low, high = 1280, 5120
            batch_size = 16
            network = 'bert-large-cased'
            max_hidden_size = binary_search_max_hidden_size(
                alg, low, high, network, batch_size)
            ips = get_ips(network, alg, batch_size, hidden_size=max_hidden_size)
            # macs, params = get_macs(
            #    network, alg, batch_size, hidden_size=max_hidden_size)

            out_file = "max_hidden_size_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_hidden_size": max_hidden_size,
                    "ips": ips,
                    #"macs": macs,
                    #"params": params,
                    "batch_size": batch_size,
                    #"TFLOPS": round(macs * ips / batch_size / 1e12, 2),
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_layer':
        for alg in algs:
            low, high = 24, 256
            batch_size = 16
            max_layer = binary_search_max_layer(alg, low, high, batch_size)
            network = 'bert-large-cased'
            ips = get_ips(network, alg, batch_size, layer_num=max_layer)
            # macs, params = get_macs(network, alg, batch_size, layer_num=max_layer)
            out_file = "max_layer_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_layer": max_layer,
                    "ips": ips,
                    # "macs": macs,
                    "batch_size": batch_size,
                    # "params": params,
                    # "TFLOPS": round(macs * ips / batch_size / 1e12, 2),
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_intermediate_size':
        for alg in algs:
            low, high = 30720, 307200
            batch_size = 16
            max_intermediate_size = binary_search_max_intermediate_size(
                alg, low, high, batch_size=batch_size)
            network = 'bert-large-cased'
            ips = get_ips(network, alg, batch_size, intermediate_size=max_intermediate_size)
            #macs, params = get_macs(network, alg, batch_size, intermediate_size=max_intermediate_size)

            out_file = "max_intermediate_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_intermediate_size": max_intermediate_size,
                    "ips": ips,
                    #"macs": macs,
                    "batch_size": batch_size,
                    #"params": params,
                    #"TFLOPS": round(macs * ips / batch_size / 1e12, 2),
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
    elif args.mode == 'swap':
        networks = ['bert-large-cased', 'bert-base-cased']
        batch_size = 16
        for network in networks:
            # original
            alg = None
            run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False)
            
            # swap unquantized tensor
            alg = 'swap'
            run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False)
            
            # swap gact 4 bit
            alg = 'L3'
            run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False)
            
    elif args.mode == 'ckpt':
        networks = ['bert-large-cased', 'bert-base-cased']
        batch_size = 16
        for network in networks:
            # original
            alg = None
            run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False)
            
            # swap unquantized tensor
            alg = 'ckpt'
            run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False)
            
            # gact fix 4 bit with gradient checkpoint
            alg = 'L1_ckpt'
            run_benchmark(network, alg, batch_size, debug_mem=True, debug_speed=False)    
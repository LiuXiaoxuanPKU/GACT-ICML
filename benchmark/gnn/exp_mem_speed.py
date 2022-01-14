import argparse
import json
import os
import time


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def alg_to_config(algorithm):
    return "--alg %s" % algorithm


def network_to_command(network):
    return "python3 test_gnn.py --model ARCH".replace("ARCH", network)


def run_benchmark(network, alg, benchmark, debug_mem=False, debug_speed=False,
                 hidden_size=None, num_layer=None):
    cmd = network_to_command(network)
    
    if benchmark == "actnn":
        cmd += f' --actnn --level {alg}'
    else:
        alg = None
        
    if hidden_size is not None:
        cmd += f' --hidden-size {hidden_size} '
    
    if num_layer is not None:
        cmd += f' --num-layers {num_layer} '
        
    if debug_speed:
         cmd += f' --get-speed '
    
    if debug_mem:
        cmd += f' --get-mem '

    ret_code = run_cmd(cmd)

    if ret_code != 0:
        out_file = "speed_results.json"
        with open(out_file, "a") as fout:
            val_dict = {
                "network": network,
                "algorithm": alg,
                "benchmark": benchmark,
                "hidden_size": hidden_size,
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


def binary_search_max_hidden_size(network, alg, benchmark, low, high):
    ret = 0
    low, high = round_up(low), round_down(high)
    while low <= high:
        mid = round_down(low + (high - low) // 2)
        success = run_benchmark(network, alg, benchmark, hidden_size=mid, debug_speed=True) == 0
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def binary_search_max_input_size(alg, low, high, network):
    ret = 0
    low, high = round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        success = (run_benchmark(network, alg, input_size=mid, debug_speed=True) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret


def binary_search_max_layer(network, alg, benchmark, low, high, num_layer):
    ret = 0
    low, high = round_up(low), round_down(high)
    while low <= high:
        mid = round_down(low + (high - low) // 2)
        success = (run_benchmark(
            network, alg, benchmark, num_layer=num_layer, debug_speed=True) == 0)
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret



def get_ips(network, alg, benchmark, hidden_size=None, num_layer=None):
    run_benchmark(network, alg, benchmark, hidden_size=hidden_size, num_layer=num_layer, debug_speed=True)
    line = list(open("speed_results.json").readlines())[-1]
    return json.loads(line)['ips']



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='binary_search_max_hidden_size')
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()

    networks = ['gcn']
    algs = ['L1', 'L1.2']
    benchmark = ['actnn']
    
    if args.mode == 'linear_scan':
        pass                  
    elif args.mode == 'binary_search_max_hidden_size':
        for network in networks:
            for alg in algs:
                for b in benchmark:
                    low, high = 1280, 12800
                    max_hidden_size = binary_search_max_hidden_size(
                        network, alg, b, low, high)
                    ips = get_ips(network, alg, b, hidden_size=max_hidden_size)

                    out_file = "max_hidden_size_results.json"
                    with open(out_file, "a") as fout:
                        val_dict = {
                            "network": network,
                            "algorithm": alg,
                            "max_hidden_size": max_hidden_size,
                            "ips": ips,
                            "tstamp": time.time()
                        }
                        fout.write(json.dumps(val_dict) + "\n")
                    print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_input_size':
        for alg in algs:
            low, high = 2560, 5120
            network = 'gcn'
            max_input_size = binary_search_max_input_size(
                alg, low, high, network, batch_size)
            ips = get_ips(network, alg, batch_size, input_size=max_input_size)

            out_file = "max_input_size_results.json"
            with open(out_file, "a") as fout:
                val_dict = {
                    "network": network,
                    "algorithm": alg,
                    "max_input_size": max_input_size,
                    "ips": ips,
                    "tstamp": time.time()
                }
                fout.write(json.dumps(val_dict) + "\n")
            print(f"save results to {out_file}")
    elif args.mode == 'binary_search_max_layer':
        for network in networks:
            for alg in algs:
                for b in benchmark:
                    low, high = 20, 400
                    max_layer = binary_search_max_layer(network, alg, b, low, high)
                    ips = get_ips(network, alg, b, layer_num = max_layer)
                    # macs, params = get_macs(network, alg, batch_size)

                    out_file = "max_layer_results.json"
                    with open(out_file, "a") as fout:
                        val_dict = {
                            "network": network,
                            "algorithm": alg,
                            "max_layer": max_layer,
                            "ips": ips,
                            # "macs": macs,
                            # "params": params,
                            # "TFLOPS": round(macs * ips / 1e12, 2),
                            "tstamp": time.time()
                        }
                        fout.write(json.dumps(val_dict) + "\n")
                    print(f"save results to {out_file}")

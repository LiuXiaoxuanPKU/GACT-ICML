import argparse
import json
import os
import time


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def network_to_command(network):
    cmd = "python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
        --cfg configs/NET_patch4_window7_224.yaml --data-path ~/imagenet --batch-size BS --get-speed "
    return cmd


def run_benchmark(network, alg, batch_size):
    cmd = network_to_command(network)
    cmd = cmd.replace("BS", f"{batch_size}")
    cmd = cmd.replace("NET", f"{network}")

    if alg != None:
        cmd += "  --level LEVEL ".replace("LEVEL", alg)


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

    time.sleep(2)
    run_cmd("nvidia-smi > /dev/null")
    time.sleep(1)
    return ret_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()

    networks = ["swin_tiny"]
    algs = [None, "L1", "L2"]
    
    for net in networks:
        for alg in algs:
            try_cnt = 0
            for batch_size in range(16, 800, 16):
                ret_code = run_benchmark(net, alg, batch_size)
                if ret_code != 0:
                    try_cnt += 1
                    if try_cnt == 3:
                        break
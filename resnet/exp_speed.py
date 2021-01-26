import os
import json
import argparse

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


os.environ['DEBUG_SPEED'] = 'true'

alg_to_config = {
    "exact": "'-c fanin'",
    "quantize": "'-c quantize --ca=True --cabits=2 --ibits=8 --calg pl'",
}

network_to_batch_size = {
    "resnet50": [160, 640],
    "resnet152": [64, 512],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['eval', 'plot'], default='eval')
    args = parser.parse_args()

    if args.mode == 'eval':
        #for network in ['resnet50']:
            #for alg in ['quantize']:
                #for batch_size in [640]:
        for network in ['resnet50', 'resnet152']:
            for alg in ['exact', 'quantize']:
                for batch_size in network_to_batch_size[network]:
                    cmd = f"bash dist-train 1 0 127.0.0.1 1 {network} " +\
                          alg_to_config[alg] + " " +\
                          f"tmp ~/imagenet {batch_size}"
                    ret_code = run_cmd(cmd)
                    if ret_code != 0:
                        break
    else:
        res_file = "mem_results.tsv"

        data = {}  # [network -> [method -> max_batch_size
        for line in open(res_file):
            val_dict = json.loads(line)


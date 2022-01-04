import traceback
import time

import subprocess


def can_train(layer, head, q_len, hidden_size, bz, actnn, efficient_softmax, grad_ckpt):
    params = ["python", "simple_run.py",
              "--layer", str(layer), "--head", str(head),
              "--q_len", str(q_len), "--hidden_size", str(hidden_size),
              "--bz", str(bz)]
    if actnn:
        params.append("--actnn")
    if efficient_softmax:
        params.append("--efficient_softmax")
    if grad_ckpt:
        params.append("--grad_ckpt")
    print(params)
    output = subprocess.run(params, capture_output=True)
    print(output)
    return output.returncode == 0


def find_max_batch_size(layer, head, q_len, hidden_size, actnn, start_bz, efficient_softmax, grad_ckpt):
    bz = start_bz
    while can_train(layer, head, q_len, hidden_size, bz, actnn, efficient_softmax, grad_ckpt):
        if bz >= 2500:
            return 2500
        bz = bz * 2

    l, r = bz//2, bz
    while l <= r:
        mid = (l + r) // 2
        if can_train(layer, head, q_len, hidden_size, mid, actnn, efficient_softmax, grad_ckpt):
            l = mid + 1
        else:
            r = mid - 1
    return r


def sweep():
    # num_bert_layers = [3, 6, 9, 12, 15, 18, 21, 24]
    # num_atten_head = [3, 6, 9, 12, 15, 18, 21, 24]
    # max_position_embeddings = [128, 256, 512]
    # hidden_sizes = [96, 192, 384, 768, 1536]
    # actnn = [False, True]

    # num_bert_layers = [3, 6, 9, 12, 15, 18, 21, 24]
    # num_atten_head = [3]
    # max_position_embeddings = [256]
    # hidden_sizes = [768]
    # actnn = [False, True]
    # efficient_softmax = [True]

    # num_bert_layers = [12]
    # num_atten_head = [3, 6, 12, 24, 48, 96]
    # max_position_embeddings = [256]
    # hidden_sizes = [768]
    # actnn = [False, True]

    # num_bert_layers = [12]
    # num_atten_head = [12]
    # max_position_embeddings = [64, 128, 256, 512, 1024]
    # hidden_sizes = [768]
    # actnn = [False, True]

    num_bert_layers = [12]
    num_atten_head = [12]
    max_position_embeddings = [512]
    hidden_sizes = [768]
    actnn = [False, True]
    efficient_softmax = [True]
    gradient_cpkt = [True, False]

    with open("results/batch_size", 'a') as f:
        f.write(
            "actnn, layer, head, q_len, hidden_size, max_batch_size, efficient_softmax\n")

    start = True
    start_bz = 2
    for layer in num_bert_layers:
        for head in num_atten_head:
            for q_len in max_position_embeddings:
                for hidden_size in hidden_sizes:
                    for e in efficient_softmax:
                        for gk in gradient_cpkt:
                            if hidden_size % head != 0:
                                continue
                            for quantize in actnn:
                                # if layer == 3 and head == 3:
                                #     start = True
                                if not start:
                                    continue
                                max_batch_size = find_max_batch_size(
                                    layer, head, q_len, hidden_size, quantize, start_bz, e, gk)
                                if not quantize:
                                    start_bz = max_batch_size
                                else:
                                    start_bz = 2
                                with open("results/batch_size", 'a') as f:
                                    f.write("%d, %d, %d, %d, %d, %d, %d, %d\n" % (
                                        quantize, layer, head, q_len, hidden_size, max_batch_size, e, gk))


if __name__ == "__main__":
    sweep()

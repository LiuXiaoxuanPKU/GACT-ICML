import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW
from transformers import get_scheduler

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from actnn.utils import get_memory_usage, compute_tensor_bytes
from actnn.controller import Controller

import time
import numpy as np
import pickle
import traceback
from pytorch_memlab import LineProfiler, MemReporter

raw_datasets = load_dataset("imdb")
num_epochs = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model(model_name, efficient_softmax, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=batch_size
    )
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    config.num_hidden_layers = 12
    config.efficient_softmax = efficient_softmax
    config.nested_checkpoint = False

    model = AutoModelForSequenceClassification.from_config(config)
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model.to(device)
    return model, train_dataloader, optimizer, lr_scheduler

def train(bit, swap, model, train_dataloader, optimizer, lr_scheduler, get_mem, get_speed):
    if get_mem and get_speed:
        print("[Error] Cannot benchmark memory and speed at the same time")
        exit(0)
    
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    if bit == -1:
        actnn = False
    else:
        actnn = True

    controller = Controller(default_bit=bit, swap=swap, debug=False, prefetch=False)
    controller.filter_tensors(model.named_parameters())

    def pack_hook(tensor):  # quantize hook
        if tensor.requires_grad and tensor.data_ptr() not in controller.unrelated_tensors:
            torch.cuda.synchronize()
            cur_mem = torch.cuda.memory_allocated()
            max_mem = torch.cuda.max_memory_allocated()
            print("[Pack]", tensor.shape, "mem: ", cur_mem / 1024 / 1024, max_mem / 1024 / 1024, type(tensor.grad_fn))
            torch.cuda.reset_max_memory_allocated()
        return tensor
        return controller.quantize(tensor)

    def unpack_hook(tensor):  # dequantize hook
        if tensor.requires_grad and tensor.data_ptr() not in controller.unrelated_tensors:
            torch.cuda.synchronize()
            cur_mem = torch.cuda.memory_allocated()
            max_mem = torch.cuda.max_memory_allocated()
            print("[Unpack] mem: ", tensor.shape, cur_mem / 1024 / 1024, max_mem / 1024 / 1024, type(tensor.grad_fn))
            torch.cuda.reset_max_memory_allocated()
        return tensor
        output = controller.dequantize(tensor)
        if tensor[0]:
            torch.cuda.synchronize()
            cur_mem = torch.cuda.memory_allocated()
            max_mem = torch.cuda.max_memory_allocated()
            print("[Unpack] mem: ", cur_mem / 1024 / 1024, max_mem / 1024 / 1024, output.shape)
            torch.cuda.reset_max_memory_allocated()
        return output

    model.train()
    if actnn:
        torch._C._autograd._register_saved_tensors_default_hooks(pack_hook, unpack_hook)

    ips_list = []
    for _ in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            batch = {k: v for k, v in batch.items()}
            batch["labels"] = batch["label"].to(device)
            del batch["label"]
            del batch["text"]

            batch_size = batch["input_ids"][0].shape[0]
            batch["input_ids"] = (
                torch.cat(batch["input_ids"], 0).reshape(batch_size, -1).to(device)
            )
            batch["attention_mask"] = (
                torch.cat(batch["attention_mask"], 0).reshape(batch_size, -1).to(device)
            )
            if "token_type_ids" in batch:
                batch["token_type_ids"] = (
                    torch.cat(batch["token_type_ids"], 0)
                    .reshape(batch_size, -1)
                    .to(device)
                )

            end = time.time()
            if get_mem:
                print("===============After Data Loading================")
                data_size = compute_tensor_bytes(
                    [
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                        batch["labels"],
                    ]
                )
                print("Data size %f MB" % (data_size / 1024 / 1024))
                # weight + optimizer state + data size
                init_mem = get_memory_usage(True)
            
            outputs = model(**batch)
            loss = outputs.loss

            if get_mem:
                print("===============Before Backward===================")
                # weight + optimizer state + data size + activation + loss + output
                before_backward = get_memory_usage(True)
                activation = (
                    before_backward
                    - init_mem
                    - compute_tensor_bytes([loss, outputs.logits])
                )

            loss.backward()
            del loss
            del outputs

            if actnn:
                controller.iterate()

            if get_mem:
                print("===============After Backward====================")
                after_backward = get_memory_usage(True)  # init + grad
                # init: weight + optimizer state + data size
                # before backward: weight + optimizer state + data size + activation + loss + output 
                # after backward: weight + optimizer state + data size + grad
                # total: weight + optimizer state + data size + activation + loss + output + grad

                # total - activation - init: loss + output + grad
                # after_backward = init + grad
                # total: weight + optimizer state + data size + activation + loss + output + grad
                # total - activation =?= after_backward
                total_mem = before_backward + (after_backward - init_mem)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            del batch

            if get_mem:
                print("Activation Mem: %d MB" % (activation / 1024 / 1024))
                print("Total Mem: %d MB" % (total_mem / 1024 / 1024))
                print(
                    "Gradient Mem: %d MB" % ((after_backward - init_mem) / 1024 / 1024)
                )
                peak_mem = torch.cuda.max_memory_allocated()
                print("Peak Mem: %d MB" % (peak_mem / 1024 / 1024))
                break

            if get_speed:
                ips_list.append(batch_size / (time.time() - end))
                end = time.time()
                if i == 4:
                    break
    if actnn:
        torch._C._autograd._reset_saved_tensors_default_hooks()

    return np.median(ips_list)

def get_model_and_train(bit, swap, efficient_softmax, 
                        batch_size, gradient_cpkt, 
                        get_mem, get_speed):
    
    model, train_dataloader, optimizer, lr_scheduler = get_model(
            "bert-base-uncased", efficient_softmax, batch_size)

    if gradient_cpkt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    ips = train(bit, swap, 
                model, train_dataloader, optimizer, lr_scheduler,
                get_mem=get_mem, get_speed=get_speed)
    return ips

def find_max_batch(bit, swap, efficient_softmax, gradient_cpkt):
    # batch_sizes = [4, 8, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]
    batch_sizes = [i * 8 for i in range(1, 50)]
    speeds = []
    for bz in batch_sizes:
        try:
            print("gradient ckpt", gradient_cpkt)
            ips = get_model_and_train(bit, swap, efficient_softmax,
                                      bz, gradient_cpkt, get_mem=False, get_speed=True)
            speeds.append(ips)
        except:
            print(traceback.format_exc())
            torch._C._autograd._reset_saved_tensors_default_hooks()
            speeds.append(-1)
            break
    speeds += (len(batch_sizes) - len(speeds)) * [-1]
    return batch_sizes, speeds

if __name__ == "__main__":
    profile_line = True
    compare_memory = False
    find_batch = False

    if profile_line:
        bit = -1
        swap = False
        batch_size = 15
        gradient_cpkt = False
        efficient_softmax = False

        get_model_and_train(bit, swap, efficient_softmax, batch_size, gradient_cpkt, get_mem=True, get_speed=False)

    if compare_memory:
        batch_size = 12
        print("=============Original============")
        bit = -1
        get_model_and_train(bit, batch_size, False)

        print("=============Checkpoint============")
        bit = -1
        get_model_and_train(bit, batch_size, True)

        print("=============Actnn============")
        bit = 4
        get_model_and_train(bit, batch_size, False)

        print("=============Actnn + Checkpoint============")
        bit = 4
        get_model_and_train(bit, batch_size, True)

    speeds_info = {}
    if find_batch:
        print("=============Original============")
        bit = -1
        swap = False
        efficient_softmax = False
        gradient_cpkt = False
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['org'] = (batch_sizes, speeds)

        print("=============Checkpoint============")
        bit = -1
        swap = False
        efficient_softmax = False
        gradient_cpkt = True
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['ckpt'] = (batch_sizes, speeds)

        print("=============Efficient Softmax============")
        bit = -1
        swap = False
        efficient_softmax = True
        gradient_cpkt = False
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['effi'] = (batch_sizes, speeds)

        print("=============Actnn============")
        bit = 4
        swap = False
        efficient_softmax = False
        gradient_cpkt = False
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['actnn'] = (batch_sizes, speeds)

        print("=============Checkpoint + Efficient Softmax============")
        bit = -1
        swap = False
        efficient_softmax = True
        gradient_cpkt = True
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['ckpt_effi'] = (batch_sizes, speeds)

        print("=============Checkpoint + Efficient Softmax + Actnn============")
        bit = 4
        swap = False
        efficient_softmax = True
        gradient_cpkt = True
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['ckpt_effi_actnn'] = (batch_sizes, speeds)

        print("=============Checkpoint + Efficient Softmax + Actnn + Swap============")
        bit = 4
        swap = True
        efficient_softmax = True
        gradient_cpkt = True
        batch_sizes, speeds = find_max_batch(bit, swap, efficient_softmax, gradient_cpkt)
        print(batch_sizes, speeds)
        speeds_info['ckpt_effi_actnn_swap'] = (batch_sizes, speeds)


        # print("=============Actnn + Checkpoint============")
        # batch_sizes, speeds = find_max_batch(4, True)
        # print(batch_sizes, speeds)
        # speeds_info['actnn_ckpt'] = (batch_sizes, speeds)

        # with open("speed", 'wb') as f:
        #     pickle.dump(speeds_info, f)
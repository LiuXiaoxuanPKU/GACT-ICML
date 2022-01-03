import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer

from datasets import load_dataset

from actnn.controller import Controller
from actnn.utils import get_memory_usage, compute_tensor_bytes

import argparse

raw_datasets = load_dataset("imdb")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_model(layer, head, q_len, hidden_size, bz):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = q_len
    print(tokenizer)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=bz, num_workers=1
    )
    for batch in train_dataloader:
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
        break

    config = AutoConfig.from_pretrained(model_name, num_labels=2)

    config.num_hidden_layers = layer
    config.efficient_softmax = False
    config.nested_checkpoint = False
    config.num_attention_heads = head
    config.max_position_embeddings = q_len
    config.hidden_size = hidden_size

    model = AutoModelForSequenceClassification.from_config(config)
    num_training_steps = len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model.to(device)
    return model, batch, optimizer, lr_scheduler

def can_train(layer, head, q_len, hidden_size, bz, actnn, get_mem):
    before_can_train_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print("Before can train", before_can_train_mem)
    if before_can_train_mem > 200:
        exit(0)
    print("==== Test Can train ====", layer, head, q_len, bz, actnn)
    def train_loop():
        model, batch, optimizer, lr_scheduler = get_model(layer, head, q_len, hidden_size, bz)
        if get_mem:
            print("====================Init Memory==============")
            init_mem = get_memory_usage(True)
            data_size = compute_tensor_bytes(
                        [
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["token_type_ids"],
                            batch["labels"],
                        ]
                    )

        def pack_hook(tensor):  # quantize hook
            print("cur %f MB, peak %f MB" % (torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)
            print("Pack ", tensor.shape, tensor.grad_fn)
            torch.cuda.reset_peak_memory_stats()
            if actnn:
                return controller.quantize(tensor)
            return tensor

        def unpack_hook(tensor):  # dequantize hook
            if actnn:
                print("cur %f MB, peak %f MB" % (torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024), flush=True)
                ret = controller.dequantize(tensor)
                print(ret.shape)
                torch.cuda.reset_peak_memory_stats()
                return ret
            return tensor

        model.train()
        controller = None
        if actnn:
            controller = Controller(default_bit=4, swap=False, debug=False, prefetch=False)
            controller.filter_tensors(model.named_parameters())

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            for i in range(2):
                outputs, loss = None, None
                outputs = model(**batch)
                loss = outputs.loss
                if get_mem:
                    print("==================Before Backward==============")
                    before_backward = get_memory_usage(True)
                    activation = (
                        before_backward
                        - init_mem
                        - compute_tensor_bytes([loss, outputs.logits])
                    )

                loss.backward()
                del outputs
                del loss

                if actnn:
                    controller.iterate()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if get_mem:
                    print("==================After Backward==============")
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
                    grad_mem = after_backward - init_mem
                    weight_optimizer_mem = init_mem - data_size
                    break

            return total_mem, activation, data_size, grad_mem, weight_optimizer_mem

    return train_loop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12, help="number of bert layers")
    parser.add_argument("--head", type=int, default=12, help="number of attention heads")
    parser.add_argument("--q_len", type=int, default=512, help="max sentence length")
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    parser.add_argument("--bz", type=int, default=8, help="batch size")
    parser.add_argument("--actnn", action="store_true", help="turn on actnn")
    parser.add_argument("--get_mem", action="store_true", help="test memory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    MB = 1024 * 1024
    total_mem, activation, data_size, grad_mem, weight_optimizer_mem = can_train(args.layer, args.head, args.q_len, args.hidden_size, args.bz, args.actnn, args.get_mem)
    print("Total Memory %f MB\tActivation %f MB\n Data Size %f MB\tGradient Size %f MB\n Weight + Optimizer %f MB" % 
            (total_mem / MB , activation / MB, data_size / MB, grad_mem / MB, weight_optimizer_mem / MB))

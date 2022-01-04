import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer

from datasets import load_dataset

from actnn.controller import Controller
import argparse

raw_datasets = load_dataset("imdb")
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model(layer, head, q_len, hidden_size, bz, efficient_softmax):
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
            torch.cat(batch["attention_mask"], 0).reshape(
                batch_size, -1).to(device)
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
    config.efficient_softmax = efficient_softmax
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


def can_train(layer, head, q_len, hidden_size, bz, actnn, efficient_softmax, gradient_checkpoint):
    before_can_train_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print("Before can train", before_can_train_mem)
    if before_can_train_mem > 200:
        exit(0)
    print("==== Test Can train ====", layer, head, q_len, bz, actnn)

    def train_loop():
        model, batch, optimizer, lr_scheduler = get_model(
            layer, head, q_len, hidden_size, bz, efficient_softmax)

        if gradient_checkpoint:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()

        def pack_hook(tensor):  # quantize hook
            if actnn:
                return controller.quantize(tensor)
            return tensor

        def unpack_hook(tensor):  # dequantize hook
            if actnn:
                # torch.cuda.empty_cache()
                return controller.dequantize(tensor)
            return tensor

        model.train()
        controller = None
        if actnn:
            controller = Controller(
                model, bit=4, swap=False, auto_prec=False, debug=False, prefetch=False)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            for i in range(2):
                outputs, loss = None, None
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                del outputs
                del loss

                if actnn:
                    controller.iterate(None)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            return True

    train_loop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12,
                        help="number of bert layers")
    parser.add_argument("--head", type=int, default=12,
                        help="number of attention heads")
    parser.add_argument("--q_len", type=int, default=512,
                        help="max sentence length")
    parser.add_argument("--hidden_size", type=int,
                        default=768, help="hidden size")
    parser.add_argument("--bz", type=int, default=8, help="batch size")
    parser.add_argument("--actnn", action="store_true", help="turn on actnn")
    parser.add_argument("--efficient_softmax",
                        action="store_true", help="turn on efficient softmax")
    parser.add_argument("--grad_ckpt",
                        action="store_true", help="turn on gradient checkpoint")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    can_train(args.layer, args.head, args.q_len,
              args.hidden_size, args.bz, args.actnn, args.efficient_softmax, args.grad_ckpt)

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW
from transformers import get_scheduler

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from actnn import get_memory_usage, compute_tensor_bytes
from actnn.controller import Controller

raw_datasets = load_dataset("imdb")
num_epochs = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model(model_name, batch_size=8):
    print(model_name)
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
    print(config)
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


def train(bit, model, train_dataloader, optimizer, lr_scheduler):
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    iter_idx = 0
    if bit == -1:
        actnn = False
    else:
        actnn = True

    controller = Controller(default_bit=bit, swap=False, debug=False, prefetch=False)
    controller.filter_tensors(model.named_parameters())

    def pack_hook(tensor):  # quantize hook
        return controller.quantize(tensor)

    def unpack_hook(tensor):  # dequantize hook
        return controller.dequantize(tensor)

    model.train()
    if actnn:
        torch._C._autograd._register_saved_tensors_default_hooks(pack_hook, unpack_hook)

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

            print("===============After Data Loading================")
            print(batch["input_ids"].shape)
            print(batch["attention_mask"].shape)
            print(batch["token_type_ids"].shape)
            print(batch["labels"].shape)
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
            print(type(model.get_input_embeddings()))
            outputs = model(**batch)
            loss = outputs.loss

            print("===============Before Backward===================")

            before_backward = get_memory_usage(
                True
            )  # weight + optimizer state + data size + activation + loss + output
            activation = (
                before_backward
                - init_mem
                - compute_tensor_bytes([loss, outputs.logits])
            )
            loss.backward()
            del loss
            print("===============After Backward====================")
            if actnn:
                controller.iterate()

            after_backward = get_memory_usage(True)  # init + grad
            # init: weight + optimizer state + data size
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

            if i == iter_idx:
                print("Activation Mem: %d MB" % (activation / 1024 / 1024))
                print("Total Mem: %d MB" % (total_mem / 1024 / 1024))
                print(
                    "Gradient Mem: %d MB" % ((after_backward - init_mem) / 1024 / 1024)
                )
                break
    if actnn:
        torch._C._autograd._reset_saved_tensors_default_hooks()


if __name__ == "__main__":
    batch_size = 4
    model, train_dataloader, optimizer, lr_scheduler = get_model(
        "bert-base-uncased", batch_size
    )
    bit = 2
    train(bit, model, train_dataloader, optimizer, lr_scheduler)

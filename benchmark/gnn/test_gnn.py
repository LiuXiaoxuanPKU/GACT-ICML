import wandb
import torch
import time
import copy
import argparse
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

import gact
from gact import config
from gact.controller import Controller  # import gact controller
from gact.utils import get_memory_usage, exp_recorder

from utils import AverageMeter

from cogdl.datasets.ogb import OGBArxivDataset
from models import GCN, SAGE, GAT
from thop import profile
import json

wandb.init(project="gact-Graph")
parser = argparse.ArgumentParser(description="GNN (gact)")
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--model", type=str, default="gcn")
parser.add_argument("--level", type=str, default="L2.2")
parser.add_argument("--nhead", type=int, default=3)
parser.add_argument("--norm", type=str, default="batchnorm")
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--gact", action="store_true")
parser.add_argument("--get-mem", action="store_true")
parser.add_argument("--get-speed", action="store_true")
parser.add_argument("--get-macs", action="store_true")
args = parser.parse_args()

wandb.config.update(args)

quantize = args.gact
get_mem = args.get_mem
get_speed = args.get_speed

device = torch.device("cuda:0")


dataset = OGBArxivDataset()
graph = dataset[0]
graph.add_remaining_self_loops()
graph.apply(lambda x: x.to(device))
num_nodes = graph.x.shape[0]

if args.model == "gcn":
    model = GCN(
        in_feats=dataset.num_features,
        hidden_size=args.hidden_size,
        out_feats=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        norm=args.norm,
    )
elif args.model == "sage":
    model = SAGE(
        in_feats=dataset.num_features,
        out_feats=dataset.num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        norm=args.norm,
    )
elif args.model == "gat":
    model = GAT(
        in_feats=dataset.num_features,
        out_feats=dataset.num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        norm=args.norm,
        nhead=args.nhead,
    )
else:
    raise NotImplementedError
print(model)
model.to(device)

if args.get_macs:
    macs, params = profile(model, inputs=(graph, ))
    macs += graph.edge_index[1].shape[0] * (args.hidden_size * (args.num_layers - 1) + 40)
    print(f"Macs: {macs}\t Params: {params}")
    out_file = "get_macs.json"
    with open(out_file, 'w') as fout:
        fout.write(json.dumps([macs, params]))
    print(f"save results to {out_file}")
    exit()

gact.set_optimization_level(args.level)
gact.set_adapt_interval(20)
controller = Controller(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def pack_hook(tensor):  # quantize hook
    if quantize:
        return controller.quantize(tensor)
    return tensor


def unpack_hook(tensor):  # dequantize hook
    if quantize:
        return controller.dequantize(tensor)
    return tensor


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


batch_total_time = 1
train_ips_list = []
# install hook
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    total_mem = AverageMeter('Total Memory', ':.4e')
    peak_mem = AverageMeter('Peak Memory', ':.4e')
    activation_mem = AverageMeter('Activation Memory', ':.4e')
    ips = AverageMeter('IPS', ':.1f')

    best_model = None
    best_acc = 0
    patience = 0
    epoch_iter = tqdm(range(args.epochs))

    if get_mem:
       init_mem = get_memory_usage(True)
 
    for i in epoch_iter:
        
        if get_speed:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
    
        model.train()
        # compute output
        output = model(graph)
        loss = F.cross_entropy(
            output[graph.train_mask], graph.y[graph.train_mask])
        # measure accuracy and record loss
        losses.update(loss.detach().item())

        if get_mem and i > 0:
            print("===============Before Backward=======================")
            before_backward = get_memory_usage(True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if get_speed:
            # measure elapsed time
            end.record()
            torch.cuda.synchronize()
            cur_batch_time = start.elapsed_time(end) / 1000.0 # event in ms
            if i > 0:
                batch_total_time += cur_batch_time
            if i % 10 == 0 and i > 0:
                if i == 10:
                    cnt = 9
                else:
                    cnt = 10
                avg_ips = cnt * 169343 / batch_total_time
                avg_batch_time = batch_total_time / 9.0
                batch_total_time = 0
                ips.update(avg_ips)
                train_ips_list.append(avg_ips)
            if i >= 10:
                train_ips = np.median(train_ips_list)
                res = "Number of nodes: %d\tIPS: %.2f\t,Cost: %.2f ms" % (
                    num_nodes, train_ips, 1000.0 / train_ips)
                print(res, flush=True)
                exp_recorder.record("network", 'gcn')
                if args.gact:
                    exp_recorder.record("algorithm", args.level)
                else:
                    exp_recorder.record("algorithm", None)
                exp_recorder.record("num_nodes", num_nodes)
                exp_recorder.record("num_layers", args.num_layers)
                exp_recorder.record("hidden_size", args.hidden_size)
                exp_recorder.record("batch_time", avg_batch_time)
                exp_recorder.record("ips", train_ips, 2)
                exp_recorder.record("tstamp", time.time(), 2)
                exp_recorder.dump('speed_results.json')
                exit(0)

        model.eval()
        for name, child in model.named_modules():
            if isinstance(child, torch.nn.BatchNorm1d):
                child.train()
                
        with torch.no_grad():
            logits = model(graph)
            val_loss = F.cross_entropy(
                logits[graph.val_mask], graph.y[graph.val_mask]).item()
            val_acc = accuracy(logits[graph.val_mask], graph.y[graph.val_mask])

        epoch_iter.set_description(
            f"Epoch: {i}" + " val_loss: %.4f" % val_loss + " val_acc: %.4f" % val_acc)
        wandb.log({"train_loss": loss.item(),
                  "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            best_model = copy.deepcopy(model)
        else:
            patience += 1
            if patience >= args.patience:
                break

        if get_mem and i > 0:
            print("===============After Backward=======================")
            after_backward = get_memory_usage(True)  # model size
            # init : weight + optimizer state + data size
            # before backward : weight + optimizer state + data size + activation + loss + output
            # after backward : init + grad
            # grad = weight
            # total - act = weight + optimizer state + data size + loss + output + grad
            total_mem.update(before_backward + (after_backward - init_mem))
            peak_mem.update(
                torch.cuda.max_memory_allocated())
            activation_mem.update(
                before_backward - after_backward)
            break

        del loss
        del output

        def get_grad():
            model.train()
            output = model(graph)
            loss = F.cross_entropy(
                output[graph.train_mask], graph.y[graph.train_mask])
            optimizer.zero_grad()
            loss.backward()
            return loss, output

        if quantize:
            controller.iterate(get_grad)

    model = best_model
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        test_acc = accuracy(logits[graph.test_mask], graph.y[graph.test_mask])
        print("Final Test Acc:", test_acc)
        wandb.log({"test_acc": test_acc})

    if get_mem:
        print("Peak %d MB" % (peak_mem.get_value() / 1024 / 1024))
        print("Total %d MB" % (total_mem.get_value() / 1024 / 1024))
        print("Activation %d MB" % (activation_mem.get_value() / 1024 / 1024))

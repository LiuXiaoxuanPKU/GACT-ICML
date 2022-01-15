import argparse
import torch
from models import GCN
from thop import profile

from cogdl.datasets.ogb import OGBArxivDataset

parser = argparse.ArgumentParser(description="GNN (ActNN)")
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--norm", type=str, default="batchnorm")
parser.add_argument("--activation", type=str, default="relu")
args = parser.parse_args()

dataset = OGBArxivDataset()

graph = dataset[0]
graph.add_remaining_self_loops()


model = GCN(
    in_feats=dataset.num_features,
    hidden_size=args.hidden_size,
    out_feats=dataset.num_classes,
    num_layers=args.num_layers,
    dropout=args.dropout,
    activation=args.activation,
    norm=args.norm,
)

macs, params = profile(model, inputs=(graph, ))
macs += graph.edge_index[1].shape[0] * (args.hidden_size * (args.num_layers - 1) + 40)
print(macs, params)

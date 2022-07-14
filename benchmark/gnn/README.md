# Getting Started

## Requirements

- Python version >= 3.7
- PyTorch version == 1.10.1

Please follow the instructions here to install PyTorch (https://github.com/pytorch/pytorch#installation).

When PyTorch has been installed,  please install the dependencies as follows

```bash
pip install cogdl thop wandb
```

## Train GCN/GAT/SAGE on ogbn-arxiv dataset
### Benchmark accuracy
```
# train with full precision
python test_gnn.py --model ARCH

# train with GACT LEVEL
python test_gnn.py --model ARCH --gact --level LEVEL
```

The choices for ARCH are {gcn, sage, gat}

The choices for LEVEL are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}

### Benchmark memory
```
# Get the memory information with full precision
python test_gnn.py --model ARCH --get-mem

# Get the memory information with GACT LEVEL
python test_gnn.py --model ARCH --gact --level LEVEL --get-mem
```

The choices for ARCH are {gcn, sage, gat}

The choices for LEVEL are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}

### Find the biggest model with full precision/GACT
```
python exp_mem_speed.py --mode binary_search_max_hidden_size
python exp_mem_speed.py --mode binary_search_max_layer
```

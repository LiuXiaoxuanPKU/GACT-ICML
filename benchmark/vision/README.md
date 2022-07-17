# Vision 
This benchmark is modified from [MONeT](https://github.com/utsaslab/MONeT/blob/master/examples/imagenet.py). 

## Requirements
Make sure you have PyTorch 1.10 and GACT installed.

1. Install the dependency
```bash
pip install thop
```
2. put imagenet dataset under `~/imagenet`.

## Benchmark ReNet on ImageNet
```
python3 train.py --data ~/imagenet --arch ARCH --batch-size BZ --get_mem
```
The choices for `ARCH` can be get with `python train.py -h`. In the paper, we tested with `resnet50`, `resnet152`.

The choices for `BZ` are the batch size you want to bechmark.

```
python3 train.py --data ~/imagenet --arch ARCH --batch-size BZ --get_mem --benchmark gact --alg LEVEL 
```
The choices for `LEVEL` are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}.


### Benchmark Speed 
```
python3 train.py --data ~/imagenet --arch ARCH --batch-size BZ --get_speed
```
The choices for `ARCH` can be get with `python train.py -h`. In the paper, we tested with `resnet50`, `resnet152`.

The choices for `BZ` are the batch size you want to bechmark.

```
python3 train.py --data ~/imagenet --arch ARCH --batch-size BZ --get_speed --benchmark gact --alg LEVEL 
```
The choices for `LEVEL` are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}.

### Find the biggest model/bacth size with full precision/GACT
Find the max bacth size
```
python exp_mem_speed.py  --mode binary_search_max_batch
```
Find the max input solution
```
python exp_mem_speed.py  --mode binary_search_max_input_size
```
Find the max model depth
```
python exp_mem_speed.py  --mode binary_search_max_layer
```
Find the max model width
```
python exp_mem_speed.py  --mode binary_search_max_width
```

### Other
Lastly, you can use the following command to sweep different optmization levels and different batch sizes.
```bash
python exp_mem_speed.py --mode linear_scan 
```
The results will be stored in `speed_results.json`.
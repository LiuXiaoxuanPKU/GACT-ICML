# Swin Transformer
This benchmark is modified from [Swin Transformer official repo](https://github.com/microsoft/Swin-Transformer) commit `6bbd83ca617db8480b2fb9b335c476ffaf5afb1a`. 

## Requirements
Make sure you have PyTorch 1.10 and GACT installed.

1. Install the dependency for Swin Transformer
```bash
pip install timm opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```
2. put imagenet dataset under `~/imagenet`.

## Benchmark Swin on ImageNet dataset
### Benchmark Memory
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
	--cfg CONFIG --data-path ~/imagenet --batch-size BZ --get-mem --level LEVEL
```
The choices for CONFIG are the config file path under `./configs`. In the paper, we tested with `configs/swin_small_patch4_window7_224.yaml`.

The choices for `BZ` are the batch size you want to bechmark.

The choices for `LEVEL` are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}.

### Benchmark Speed
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
	--cfg CONFIG --data-path ~/imagenet --batch-size BZ --get-speed --level LEVEL
```
The choices for CONFIG are the config file path under `./configs`. In the paper, we tested with `configs/swin_small_patch4_window7_224.yaml`.

The choices for `BZ` are the batch size you want to bechmark.

The choices for `LEVEL` are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}.

### Other
You can use the following command to sweep different optmization levels and different batch sizes for swin_tiny.
```bash
python exp_mem_speed.py --mode linear_scan 
```
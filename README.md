INSTALL
====
Tested with PyTorch 1.6 + CUDA 10.2

Step 1: Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Step 2: Install this repo
```bash
nvcc --version   # Should be 10.2
cd quantize
pip install -e .
```

Using the library
====

```python
from quantize import QConv2d, QLinear, QBatchNorm2d, QModule

class MyModel(nn.Module):
    # Replace all Conv2d layers to QConv2d layers
    def __init__(self): 
        self.conv = QConv2d(...)
        self.linear = QLinear(...)
        self.bn = QBatchNorm2d(...)

# Convert the model to a QModule before using 
model = QModule(MyModel())  
```


Using the built-in ResNet examples
====

Prepare
```bash
cd resnet
wget https://people.eecs.berkeley.edu/~jianfei/results.tar.gz
tar xzvf results.tar.gz
mkdir results/tmp
```


CIFAR10
----

```
# Testing
python main.py <data config> <quantize config> --workspace results/tmp --evaluate --training-only --resume results/cifar10/checkpoint-10.pth.tar --resume2 results/cifar10/checkpoint-10.pth.tar  ~/data/cifar10
# Training
python main.py <data config> <quantize config> --workspace results/tmp --gather-checkpoints  ~/data/cifar10
```
where data config is 

```--dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0```

| *quantize config* | *Overall Var* | *Val Top1* |
|--------|----------|---------|
| -c fanin | /  | 92.78 |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.011829948052763939 | 93.10 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.29687657952308655 | 91.58 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.03139203414320946 | 93.07 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.0061396025121212006 | 92.87 | 
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False --usegradient=False | 0.033477190881967545 | 92.67 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.0026178082916885614 | 92.91 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True --usegradient=False | 0.019806843250989914 | 93.24 |

CIFAR100
----

```
# Testing
python main.py <data config> <quantize config> --workspace results/tmp --evaluate --training-only --resume results/cifar100/checkpoint-10.pth.tar --resume2 results/cifar100/checkpoint-10.pth.tar  ~/data/cifar100
# Training
python main.py <data config> <quantize config> --workspace results/tmp --gather-checkpoints  ~/data/cifar100
```
where data config is

```--dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0```

| *quantize config* | *Overall Var* | *Val Top1* |
|--------|----------|---------|
| -c fanin | /  | 70.33 |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.03873932361602783 | 69.87 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.9092444181442261 | 61.55 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.1514340490102768 | 69.81 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.08524089306592941 | 70.40 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False --usegradient=False | 0.15131664276123047 | 70.14 | 
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.032878682017326355 | 70.36 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True --usegradient=False | 0.0763486847281456 | 69.91 | 

ImageNet
----

```
# Testing
./dist-test 1 0 127.0.0.1 1 resnet50 "<quantize config>" tmp <imagenet path>
# Training
./dist-train 1 0 127.0.0.1 8 resnet50 "<quantize config>" tmp <imagenet path> 
```

| *quantize config* | *Overall Var* | *Val Top1* |
|--------|----------|---------|
| -c fanin | /  | 77.09 |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.09732577949762344 | ~~77.40， g29~~ |
| qlinear, 2bit, persample minimax |/ | 75.54 |
| qlinear, 2bit, persample minimax, per sample adaptive |/ | 75.93 |
| qlinear, 2bit, persample minimax, per layer |/ | 76.35 |
| qlinear, 4bit |/ | 76.81 |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.8921149969100952 | 76.57 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 47.97850799560547 | |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.7865011692047119 | ~~77.06~~ 0.1? |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.5164923667907715 |   |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False --usegradient=False | 0.7651944756507874 |  |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True |0.12191561609506607  | 76.92 | 
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True --usegradient=False | 0.2176554799079895 | 77.06 |

Coco
----

| *quantize config* | *Quant Var* | *Sample Var* |
|--------|----------|---------|
| QBN, -c quantize --ca=True --cabits=4 --ibits=8 --pergroup=True --perlayer=False --usegradient=False | 19 |  |
| QBN, -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False --usegradient=False |  |  |
| QBN, -c quantize --ca=True --cabits=3 --ibits=8 --pergroup=True --perlayer=True --usegradient=False |  12.8 |  | 
| QBN, -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True --usegradient=False |  127.8656 |  |
| QSyncBN, -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False --usegradient=False | 493.7278 | | 
| QSyncBN, -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True --usegradient=False | 123  | 5836.6851 |

TODOs
----

0. ~~Better API~~ [Jianfei, today]
1. ~~True quantize~~ [Lianmin, next Monday]
2. Speed optimization [Lianmin]
3. Depthwise [Jianfei]
4. Paper writing [Jianfei, next next Thursday]
5. ~~Verification experiments~~ [Dequan, this year]
6. Result for detection [Dequan]
7. 10 machine -> 1 machine [Dequan]
8. Pool, Upsample [Lianmin]
9. SyncBN [Jianfei]
10. ~~Fix BatchNorm~~ [Jianfei, today]
11. ~~CIFAR100 results~~
12. float16 / QAT? 

*. Deadline [Feb. 7]

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
cd pytorch_minimax
python setup.py install
cd ..
cd quantize
pip install -e .
cd ..
```

Memory Saving Training
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
| -c qlinear --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.009703520685434341 | |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.21035686135292053 | |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.0181496012955904 | |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.002921732608228922 | |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.001438044011592865 | |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.011829948052763939 | |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.29687657952308655 | |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.023775238543748856 | |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.0034970948472619057 | | 
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.0017123270081356168 |  |

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
| -c qlinear --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.03486475348472595 | |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.7869864702224731 | |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.09328116476535797 | |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.04863186180591583 | |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.018564047291874886 | |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.03873932361602783 | |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 0.9092444181442261 | |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.1020037978887558 | |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.05299000069499016 |  | 
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.0214 | |

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
| -c qlinear --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.7898967862129211 | |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 9.28708267211914 | |
| -c qlinear --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.3763839602470398 | |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.22041866183280945 | |
| -c qlinear --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.05691418796777725 | g15 |
| qlinear, 2bit, persample minimax |/ | 75.54 |
| qlinear, 2bit, persample minimax, per sample adaptive |/ | 75.93 |
| qlinear, 4bit |/ | 76.81 |
| -c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False | 0.8921149969100952 | g36 |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False | 47.97850799560547 | |
| -c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False | 0.40297362208366394 | g25 |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False | 0.25603559613227844 |   |
| -c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True | 0.06779336929321289 | g16 |

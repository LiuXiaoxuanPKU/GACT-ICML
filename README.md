INSTALL
====
Tested with PyTorch 1.4.0 + CUDA 10.1.

Step 1: Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Step 2: Install this repo
```bash
nvcc --version   # Should be 10.1
cd pytorch_minimax
python setup.py install
cd ..
cd quantize
python setup.py install
cd ..
export PYTHONPATH=$PWD:$PYTHONPATH
```

CIFAR10
====

```bash
mkdir results
# Exact
./train_cifar preact_resnet56 exact "" 
# QAT
./train_cifar preact_resnet56 qat "-c quantize --qa=True --qw=True --qg=False" 
# 8-bit PTQ
./train_cifar preact_resnet56 ptq_8 "-c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False --bbits=8"
# 8-bit PSQ 
./train_cifar preact_resnet56 psq_8 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits=8" 
# 8-bit BHQ
./train_cifar preact_resnet56 hqq_8 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits=8"   
```

ImageNet
====

Tested with GTX 1080Ti * 8, RTX 2080Ti * 8, and V100 * 8.
```bash
mkdir results
# Exact
./dist-train 1 0 127.0.0.1 8 resnet18 "" exact <imagenet_path>  
# QAT
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=False" qat <imagenet_path>  
# 8-bit PTQ
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False --bbits=8" ptq_8 <imagenet_path>
# 8-bit PSQ
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits=8" psq_8 <imagenet_path> 
# 8-bit BHQ
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits=8" bhq_8 <imagenet_path>
```

Memory Saving Training
====

Quick test
```bash
cd resnet
wget https://github.com/cjf00000/RN50v1.5/releases/download/v0.1/results.tar.gz
tar xzvf results.tar.gz
python main.py --dataset cifar10 --gather-checkpoints --arch preact_resnet56 --gather-checkpoints --workspace results/exact --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 --ca=True -c quantize --qa=False --qw=False --qg=False --biprecision=False --resume results/exact/checkpoint-1.pth.tar --epochs 2 --evaluate  -j 0 --training-only   ~/data/cifar10		# Overall var should be around 0.947
```

Full training
```
mkdir results/ca2
python main.py --dataset cifar10 --gather-checkpoints --arch preact_resnet56 --gather-checkpoints --workspace results/ca2 --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 --ca=True -c quantize --qa=False --qw=False --qg=False --biprecision=False --cabits=2   -j 0  ~/data/cifar10
```

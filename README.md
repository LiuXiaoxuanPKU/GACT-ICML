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

Quick test
```bash
cd resnet
wget https://github.com/cjf00000/RN50v1.5/releases/download/v0.1/results.tar.gz
tar xzvf results.tar.gz
python main.py --dataset cifar10 --gather-checkpoints --arch preact_resnet56 --gather-checkpoints --workspace results/exact --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 --ca=True --cabits=2 -c quantize --resume results/exact/checkpoint-1.pth.tar --epochs 2 --evaluate  -j 0 --training-only   ~/data/cifar10		# Overall var should be around 0.084
```

Full training
```
mkdir results/ca2
python main.py --dataset cifar10 --gather-checkpoints --arch preact_resnet56 --gather-checkpoints --workspace results/ca2 --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 --ca=True --cabits=2 -c quantize -j 0  ~/data/cifar10
```

# Image Classficiation

## Requirement
- Put the ImageNet dataset to `~/imagenet`
- Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Train resnet56 on cifar10
```
mkdir -p results/tmp
python3 main.py --dataset cifar10 --arch preact_resnet56 --epochs 200 --num-classes 10 \
  -j 0 --weight-decay 1e-4 --batch-size 128 --label-smoothing 0 \
  --lr 0.1 --momentum 0.9  --warmup 4 \
  -c quantize --ca=True --cabits=2 --ibits=8 --calg pl \
  --workspace results/tmp --gather-checkpoints  ~/data/cifar10
```

## Train resnet50 on imagenet
```
./dist-train 1 0 127.0.0.1 1 resnet50 \
   "-c quantize --ca=True --cabits=2 --ibits=8 --calg pl"\
   tmp ~/imagenet 256
```

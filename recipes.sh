# Environment: python 3.6 + (anaconda) pytorch 1.3.1 + nvidia apex

mkdir results/
./train resnet18 
./train resnet18 "-c quantize --persample=True"
# ./test results/resnet50 "-c quantize"
./train_cifar preact_resnet56 preact_resnet56
./train_cifar preact_resnet56 preact_resnet56 "-c quantize --persample=True"
CUDA_VISIBLE_DEVICES=0,1 ./test_cifar 2 preact_resnet56 200 "-c quantize --persample=True"

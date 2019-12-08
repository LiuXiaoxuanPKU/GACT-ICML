# ./test results/resnet50 "-c quantize"
CUDA_VISIBLE_DEVICES=1,6 ./test_cifar 2 preact_resnet20 200 "-c quantize --persample=True"

./train_cifar preact_resnet20
./train_cifar preact_resnet56
./train_cifar preact_resnet110
./train_cifar preact_resnet164
./train_cifar preact_resnet1001


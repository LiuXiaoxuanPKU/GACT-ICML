import os

# for bbits in [3, 4, 5, 6, 7, 8]:
#   for persample in [False, True]:
#     for biased in [False, True]:
for bbits, persample, biased in [
  (8, False, False),
  (6, False, False),
  (4, False, False),
  (4, True, False),
  (4, True, True),
]:
      cmd = 'CUDA_VISIBLE_DEVICES=0,1 ./test_cifar 2 preact_resnet56 200 "-c quantize --bbits {bbits} --qa False --persample={persample} --biased {biased} --qw False --batch-size 50" 29500 | tee b{bbits2}_p{persample}_bias{biased}.log'.format(
        bbits=bbits+1 if biased else bbits, persample=persample, biased=biased, bbits2=bbits
      )
      print(cmd)
      os.system(cmd)

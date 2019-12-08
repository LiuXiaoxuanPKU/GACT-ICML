import sys
import json

with open(sys.argv[1]) as f:
    data = json.load(f)

data = data['epoch']
for epoch, val, ips in zip(data['ep'], data['val.top1'], data['train.total_ips']):
    print('Epoch {}, val.top1 {}, train.total_ips {}'.format(epoch, val, ips))

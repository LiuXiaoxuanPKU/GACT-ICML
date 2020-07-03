import sys
import json

with open(sys.argv[1]) as f:
    data = json.load(f)

data = data['epoch']
for epoch, val, vall, train, trainl, ips in zip(data['ep'], data['val.top1'], data['val.loss'],
                                 data['train.top1'], data['train.loss'], data['train.total_ips']):
    print('Epoch {}, val.top1 {}, val.loss {}, train.top1 {}, train.loss {}, train.total_ips {}'.format(epoch, val, vall, train, trainl, ips))

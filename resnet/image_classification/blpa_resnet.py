'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

cnt = 0

class BLPABottleneck(nn.Module):
    expansion = 4
    M = 3

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BLPABottleneck, self).__init__()
        self.bn1 = builder.batchnorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = builder.conv1x1(inplanes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn3 = builder.batchnorm(planes, last_bn=True)
        self.conv3 = builder.conv1x1(planes, planes*4)
        self.downsample = downsample
        self.stride = stride
        self.debug = False

    def forward(self, x):
        global cnt
        cnt += 1
        residual = x

        if self.downsample is not None and self.stride == 2:
            residual = self.downsample(x)

        # print('layer ', cnt, 'in1', x[0].sum())
        out = self.bn1(x)
        # print('layer ', cnt, 'bn1', out[0].sum())
        out = self.relu(out)
        # print('layer ', cnt, 'relu1', out[0].sum())

        if self.downsample is not None and self.stride == 1:
            residual = self.downsample(out)

        out = self.conv1(out)
        # print('layer ', cnt, 'conv1', out[0].sum(), out.shape)

        cnt += 1
        out = self.bn2(out)
        # print('layer ', cnt, 'bn2', out[0].sum())
        out = self.relu(out)
        # print('layer ', cnt, 'relu2', out[0].sum())
        out = self.conv2(out)
        # print('layer ', cnt, 'conv2', out[0].sum(), out.shape)

        cnt += 1
        out = self.bn3(out)
        # print('layer ', cnt, 'bn3', out[0].sum())
        out = self.relu(out)
        # print('layer ', cnt, 'relu', out[0].sum())
        out = self.conv3(out)
        # print('layer ', cnt, 'conv3', out[0].sum(), out.shape)

        out += residual

        return out


class BLPAResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, num_classes=10):
        super(BLPAResNet, self).__init__()
        self.inplanes = 16
        self.builder = builder
        self.conv1 = builder.conv3x3(3, 16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn = builder.batchnorm(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = builder.linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            plane_diff = planes * block.expansion - self.inplanes
            print('Creating downsample layer ', stride, self.inplanes, planes * block.expansion, plane_diff)

            def downsample(out):
                out = F.avg_pool2d(out, stride)
                zero = torch.zeros([out.shape[0], plane_diff//2, out.shape[2], out.shape[3]],
                                   layout=out.layout, dtype=out.dtype, device=out.device)
                out = torch.cat([zero, out, zero], 1)
                return out


        layers = []
        layers.append(block(self.builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.builder, self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x[0].sum())
        # print(x[0])
        x = self.conv1(x)
        # print(x[0].sum())
        # print(x[0])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # exit(0)

        return x

    def set_name(self):
        self.linear_layers = [self.conv1]
        self.conv1.layer_name = 'conv_0'
        for lid, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            for bid, block in enumerate(layer):
                for cid, convlayer in enumerate([block.conv1, block.conv2, block.conv3]):
                    convlayer.layer_name = 'conv_{}_{}_{}'.format(lid+1, bid+1, cid+1)
                    self.linear_layers.append(convlayer)

        self.fc.layer_name = 'fc'
        self.linear_layers.append(self.fc)


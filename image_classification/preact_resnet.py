'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math
import torch
import torch.nn as nn


class PreActBlock(nn.Module):
    expansion = 1
    M = 2

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(PreActBlock, self).__init__()
        self.bn1 = builder.batchnorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.conv2 = builder.conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4
    M = 3

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = builder.batchnorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn2 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn3 = builder.batchnorm(planes, last_bn=True)
        self.conv3 = builder.conv1x1(planes, planes*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class PreActResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.builder.conv1x1(self.inplanes, planes * block.expansion, stride=stride)
            )

        layers = []
        layers.append(block(self.builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.builder, self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

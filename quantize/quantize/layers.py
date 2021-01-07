# The code is compatible with PyTorch 1.6

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from quantize.ops import linear, batch_norm, conv2d, sync_batch_norm, ext_quantization
from quantize.qscheme import QScheme
from quantize.qbnscheme import QBNScheme
from quantize.conf import config


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.scheme = QScheme(num_locations=kernel_size**2)

    def forward(self, input):
        if config.training:
            if self.padding_mode != 'zeros':
                return conv2d().apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                      self.weight, self.bias, self.stride,
                                      _pair(0), self.dilation, self.groups, self.scheme)
            return conv2d().apply(input, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups, self.scheme)
        else:
            return super(QConv2d, self).forward(input)


class QLinear(nn.Linear):
    num_layers = 0

    def __init__(self, input_features, output_features, bias=True):
        super(QLinear, self).__init__(input_features, output_features, bias)
        self.scheme = QScheme()

    def forward(self, input):
        if config.training:
            return linear().apply(input, self.weight, self.bias, self.scheme)
        else:
            return super(QLinear, self).forward(input)


class QBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(QBatchNorm2d, self).__init__(num_features)
        self.scheme = QBNScheme()
        # self.scheme.initial_bits = self.scheme.bits # TODO hack

    def forward(self, input):
        if not config.training:
            return super(QBatchNorm2d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return batch_norm().apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)


class QReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ext_quantization.act_quantized_relu(input)


class QSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group=None
    ) -> None:
        super(QSyncBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group)
        self.scheme = QBNScheme()

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None

        need_sync = bn_training
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return batch_norm().apply(
                input, running_mean, running_var, self.weight, self.bias,
                bn_training, exponential_average_factor, self.eps, self.scheme)
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            assert bn_training
            return sync_batch_norm().apply(
                input, self.weight, self.bias, running_mean, running_var,
                self.eps, exponential_average_factor, process_group, world_size, self.scheme)

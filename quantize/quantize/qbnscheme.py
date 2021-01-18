import torch

from quantize.conf import config
from quantize.qscheme import QScheme
from quantize.ops import ext_minimax
from C import calc_precision_dp, calc_precision, calc_avg_bits


class QBNScheme(QScheme):
    layers = []

    def __init__(self, group=0):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits[group]
        QBNScheme.layers.append(self)
        if len(QScheme.layers) > 0:
            self.prev_linear = QScheme.layers[-1]
        else:
            self.prev_linear = None

    def compute_quantization_bits(self, input):
        N, D, H, W = input.shape
        input_flatten = input.view(N, -1)
        num_features = input_flatten.shape[1]
        num_pixels = num_features // D

        # Compute min, max by groups
        if num_features % config.group_size != 0:
            # Padding
            new_num_features = (num_features // config.group_size + 1) * config.group_size
            delta = new_num_features - num_features
            input_flatten = torch.cat([input_flatten,
                                       torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

        input_groups = input_flatten.view(-1, config.group_size)
        mn, mx = ext_minimax.minimax(input_groups)
        if not config.pergroup:    # No per group quantization
            mn = torch.ones_like(mn) * mn.min()
            mx = torch.ones_like(mx) * mx.max()

        # Average range over pixels [N]
        Range_sqr = torch.norm((mx - mn).view(N, -1), dim=1).square() * (config.group_size / num_pixels)

        # greedy
        C = Range_sqr.cpu()
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, C, w, int(self.bits * N))

        return input_groups.view(N, -1, config.group_size), b.cuda(), mn.view(N, -1, 1), mx.view(N, -1, 1)

    @staticmethod
    def allocate_perlayer():
        for layer in QBNScheme.layers:
            layer.bits = layer.prev_linear.bits

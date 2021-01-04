import torch
from quantize.conf import config
from quantize.qscheme import QScheme
from C import calc_precision_dp, calc_precision, calc_avg_bits
import pytorch_minimax


class QBNScheme(QScheme):
    layers = []

    def __init__(self):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits
        self.scales = torch.ones(QScheme.num_samples)
        self.term2 = 1.0
        QBNScheme.layers.append(self)
        QScheme.all_layers.append(self)
        self.prev_linear = QScheme.layers[-1]

        # debug
        # self.name = 'bn_layer_{}'.format(QBNScheme.num_layers)
        # QBNScheme.num_layers += 1

    def set_scale(self, grad_output, weight, batch_std, normalized):
        if QScheme.update_scale:
            with torch.no_grad():
                bs, C, H, W = grad_output.shape
                N =  bs * H * W
                d = (grad_output * normalized).sum([0, 2, 3])   # [C]
                w2_s2 = (weight / batch_std) ** 2               # [C]
                w2_s4 = w2_s2 / batch_std**2
                w4_s4 = (w2_s2 ** 2).view(1, -1, 1, 1)
                term1 = (N * w4_s4 * grad_output**2).mean([1, 2, 3]) * C   # [N]
                term2 = (w2_s4 * d).sum() ** 2             # []

                term1 = term1.cpu()
                term2 = term2.cpu()

                if config.use_gradient:
                    assert QScheme.batch is not None
                    self.scales[QScheme.batch] = term1
                    self.term2 = term2
                else:
                    self.scales = term1.mean()
                    self.term2 = term2

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
        mn = pytorch_minimax.min(input_groups).view(N, -1, 1)  # N, num_groups, 1
        mx = pytorch_minimax.max(input_groups).view(N, -1, 1)  # N, num_groups, 1
        if not config.pergroup:    # No per group quantization
            min_scalar = mn.min()
            mn = torch.ones_like(mn) * min_scalar
            max_scalar = mx.max()
            mx = torch.ones_like(mx) * max_scalar

        # Average range over pixels [N]
        Range_sqr = ((mx - mn) ** 2).view(N, -1).sum(1) * config.group_size / num_pixels

        # greedy
        term1 = self.get_scale().cuda()
        term2 = self.term2
        C = (Range_sqr * (term1 * num_pixels + term2)).cpu()

        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, C, w, int(self.bits * N))

        return input_groups.view(N, -1, config.group_size), b, mn, mx

    @staticmethod
    def allocate_perlayer():
        for layer in QBNScheme.layers:
            layer.bits = layer.prev_linear.bits

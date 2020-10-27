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
        self.b = None
        self.batch_std = 1.0
        self.conv_input_norm = torch.tensor(1.0)
        self.mean_grad_norm = 1.0
        QBNScheme.layers.append(self)
        self.prev_linear = QScheme.layers[-1]

        # debug
        # self.name = 'bn_layer_{}'.format(QBNScheme.num_layers)
        # QBNScheme.num_layers += 1

    def set_scale(self, grad, batch_std, mean_grad_norm):
        if QScheme.update_scale:
            assert QScheme.batch is not None
            with torch.no_grad():
                scale = (grad.view(grad.shape[0], -1) ** 2).sum(1).detach().cpu()
                self.scales[QScheme.batch] = scale
                self.batch_std = batch_std
                self.mean_grad_norm = mean_grad_norm

    def compute_quantization_bits(self, input):
        N, num_channels, H, W = input.shape
        input_flatten = input.view(N, -1)

        # greedy
        grad_sum = self.get_scale().cuda()
        mn = pytorch_minimax.min(input_flatten)
        mx = pytorch_minimax.max(input_flatten)
        Range = mx - mn

        H_s = (input / self.batch_std) ** 2
        conv_input_norm = self.prev_linear.conv_input_norm
        coef_1 = self.mean_grad_norm * conv_input_norm
        coef_2 = H_s.sum() * conv_input_norm.sum() * grad_sum / num_channels / (N*H*W)**2

        C = (Range ** 2 * (coef_1 + coef_2) / 4).cpu()
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, C, w, int(self.bits * N))

        if config.simulate:
            mn = mn.unsqueeze(1)
            mx = mx.unsqueeze(1)

        return b, mn, mx

    @staticmethod
    def allocate_perlayer():
        for layer in QBNScheme.layers:
            layer.bits = layer.prev_linear.bits

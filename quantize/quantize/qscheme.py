import torch
import pytorch_minimax

from quantize.conf import config
from C import calc_precision_dp, calc_precision, calc_avg_bits


class QScheme(object):
    num_samples = 0
    num_layers = 0
    batch = None
    update_scale = True
    layers = []

    def __init__(self, num_locations=1):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits
        assert QScheme.num_samples > 0
        self.scales = torch.zeros(QScheme.num_samples)
        QScheme.layers.append(self)
        self.C = None
        self.dim = None
        self.num_locations = num_locations
        self.conv_input_norm = torch.tensor(1.0)

        # debug
        self.name = 'layer_{}'.format(QScheme.num_layers)
        QScheme.num_layers += 1

    def get_scale(self):
        assert QScheme.batch is not None
        scale = self.scales[QScheme.batch].clone()
        avg_scale = scale.mean()
        scale[scale == 0] = avg_scale + 1e-9
        return scale

    def set_scale(self, grad):
        if QScheme.update_scale:
            assert QScheme.batch is not None
            scale = (grad.view(grad.shape[0], -1) ** 2).sum(1).detach().cpu()
            self.scales[QScheme.batch] = self.scales[QScheme.batch] * 0.5 + scale * 0.5

    def compute_quantization_bits(self, input):
        N = input.shape[0]
        D = input.shape[1]

        R = len(input.shape)
        ranks = range(R)
        group_dims = [i if i != -1 else R - 1 for i in config.group_dims]
        reduce_dims = list(filter(lambda i: not (i in group_dims), ranks))

        # Compute the range by groups
        mn = input
        mx = input
        for dim in reduce_dims:
            mn = torch.min(mn, dim, keepdim=True)[0]
            mx = torch.max(mx, dim, keepdim=True)[0]

        input_flatten = input.view(N, -1)

        # greedy
        grad_sum = self.get_scale().cuda()

        mn2 = pytorch_minimax.min(input_flatten)
        mx2 = pytorch_minimax.max(input_flatten)
        Range = mx2 - mn2
        C = (self.num_locations * D / 4 * Range ** 2 * grad_sum).cpu()
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, C, w, int(self.bits * N))

        with torch.no_grad():
            self.C = C
            self.dim = input.numel() // N
            self.b = b.detach()
            self.conv_input_norm = (input_flatten ** 2).sum(1) * self.num_locations

        return b, mn, mx

    @staticmethod
    def allocate_perlayer():
        layers = QScheme.layers
        L = len(layers)

        if config.activation_compression_bits == config.initial_bits:
            C = torch.tensor([layer.C.sum() for layer in layers])
            w = torch.tensor([layer.dim for layer in layers], dtype=torch.int)
            total_bits = w.sum() * config.activation_compression_bits
            b = torch.ones(L, dtype=torch.int32) * 8
            b = calc_precision(b, C, w, total_bits)

            # print('=====')
            # print(b)
            # print(C)
            # print(w)
            # print(C / w)
            # print('=====')

            for i in range(L):
                layers[i].bits = layers[i].initial_bits = b[i]
        else:
            Cs = [layer.C for layer in layers]
            C = torch.cat(Cs, 0)

            N = Cs[0].shape[0]

            # TODO ???
            Ws = [torch.ones(N, dtype=torch.int32) * layer.dim for layer in layers]
            # Ws = [torch.ones(N, dtype=torch.int32) for layer in layers]
            w = torch.cat(Ws, 0)

            total_bits = w.sum() * config.activation_compression_bits
            b = torch.ones(N * L, dtype=torch.int32) * config.initial_bits
            b = calc_precision(b, C, w, total_bits)
            for i in range(L):
                bs = b[i*N : (i+1)*N]
                layers[i].bits = bs.float().mean()


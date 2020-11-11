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
        self.scales = torch.ones(QScheme.num_samples)
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
        return self.scales[QScheme.batch]

    def set_scale(self, grad):
        if QScheme.update_scale:
            assert QScheme.batch is not None
            scale = (grad.view(grad.shape[0], -1) ** 2).sum(1).detach().cpu()
            # TODO new
            self.scales[QScheme.batch] = scale / scale.sum()

    def compute_quantization_bits(self, input):
        N = input.shape[0]
        D = input.shape[1]
        input_flatten = input.view(N, -1)

        # greedy
        grad_sum = self.get_scale().cuda()
        mn = pytorch_minimax.min(input_flatten)
        mx = pytorch_minimax.max(input_flatten)
        if not config.persample:
            mn = torch.ones_like(mn) * mn.min()
            mx = torch.ones_like(mx) * mx.max()

        Range = mx - mn
        C = (self.num_locations * D / 4 * Range ** 2 * grad_sum).cpu()
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, C, w, int(self.bits * N))

        with torch.no_grad():
            self.C = C
            self.dim = input.numel() // N
            self.b = b.detach()
            self.conv_input_norm = (input_flatten ** 2).sum(1) * self.num_locations

        # B = 2 ** b - 1
        # self.var = (self.C / B**2).sum()

        if config.simulate:
            mn = mn.unsqueeze(1)
            mx = mx.unsqueeze(1)

        return b, mn, mx

    @staticmethod
    def allocate_perlayer():
        layers = QScheme.layers
        L = len(layers)

        Cs = [layer.C for layer in layers]
        C = torch.cat(Cs, 0)

        N = Cs[0].shape[0]

        Ws = [torch.ones(N, dtype=torch.int32) * layer.dim for layer in layers]
        w = torch.cat(Ws, 0)

        total_bits = w.sum() * config.activation_compression_bits
        b = torch.ones(N * L, dtype=torch.int32) * config.initial_bits
        b = calc_precision(b, C, w, total_bits)
        for i in range(L):
            bs = b[i*N : (i+1)*N]
            # print(i, w[i*N], Cs[i].mean(), bs.float().mean(), bs)
            layers[i].bits = bs.float().mean()


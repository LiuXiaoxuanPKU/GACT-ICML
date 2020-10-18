import torch
from quantize.conf import config
from C import calc_precision_dp, calc_precision, calc_avg_bits
import pytorch_minimax


class QBNScheme:

    num_layers = 0
    layers = []
    update_scale = True

    def __init__(self):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits
        self.scale = 1.0
        QBNScheme.layers.append(self)

        # debug
        self.name = 'bn_layer_{}'.format(QBNScheme.num_layers)
        QBNScheme.num_layers += 1

    def set_scale(self, mean_grad):
        if QBNScheme.update_scale:
            self.scale = (mean_grad ** 2).sum().detach()

    def compute_quantization_bits(self, input):
        N, _, H, W = input.shape
        input_flatten = input.view(N, -1)

        # greedy
        grad_sum = self.scale
        mn = pytorch_minimax.min(input_flatten)
        mx = pytorch_minimax.max(input_flatten)
        Range = mx - mn
        C = H * W * Range ** 2 * grad_sum / 4
        self.C = C.cpu()
        self.dim = input.numel() // N
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, self.C, w, int(self.bits * N))
        # print(self.C, b)

        # TODO hack
        B = 2 ** b - 1
        self.var = (self.C / B**2).sum()
        self.stats = (grad_sum, (Range**2).sum(), H*W)

        if config.simulate:
            mn = mn.unsqueeze(1)
            mx = mx.unsqueeze(1)

        return b, mn, mx

    @staticmethod
    def allocate_perlayer():
        layers = QBNScheme.layers
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
            print(layers[i].name, layers[i].stats, w[i*N], Cs[i].mean(), bs.float().mean(), bs)
            layers[i].bits = bs.float().mean()
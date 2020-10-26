import torch
from quantize.conf import config
from C import calc_precision_dp, calc_precision, calc_avg_bits
import pytorch_minimax


class QBNScheme:

    num_samples = 0
    num_layers = 0
    layers = []
    batch = None
    update_scale = True

    def __init__(self):
        self.initial_bits = config.initial_bits
        self.bits = config.activation_compression_bits
        self.scales = torch.ones(QBNScheme.num_samples)
        self.b = None
        self.batch_std = 1.0
        QBNScheme.layers.append(self)

        # debug
        self.name = 'bn_layer_{}'.format(QBNScheme.num_layers)
        QBNScheme.num_layers += 1

    # def set_scale(self, mean_grad):
    #     if QBNScheme.update_scale:
    #         self.scale = (mean_grad ** 2).sum().detach()

    def get_scale(self):
        assert QBNScheme.batch is not None
        return self.scales[QBNScheme.batch]

    def set_scale(self, grad, batch_std):
        if QBNScheme.update_scale:
            assert QBNScheme.batch is not None
            scale = (grad.view(grad.shape[0], -1) ** 2).sum(1).detach().cpu()
            self.scales[QBNScheme.batch] = scale
            self.batch_std = batch_std

    def compute_quantization_bits(self, input):
        N, _, H, W = input.shape
        input_flatten = input.view(N, -1)

        # greedy
        grad_sum = self.get_scale().cuda()
        mn = pytorch_minimax.min(input_flatten)
        mx = pytorch_minimax.max(input_flatten)
        Range = mx - mn

        H_s = (input / self.batch_std) ** 2
        coef_1 = (H_s.sum((1, 2, 3)) * grad_sum).sum()
        coef_2 = H_s.sum() * grad_sum

        # print(Range)
        C = Range ** 2 * (coef_1 + coef_2) / (4 * (N*H*W)**2)
        self.C = C.cpu()
        self.dim = input.numel() // N
        b = torch.ones(N, dtype=torch.int32) * self.initial_bits
        w = torch.ones(N, dtype=torch.int32)
        b = calc_precision(b, self.C, w, int(self.bits * N))
        # if self.b is not None:
        #     b = self.b # TODO hack
            # print(b)
        # print(self.C, b)

        # TODO hack
        B = 2 ** b - 1
        self.var = (self.C / B**2).sum()
        self.stats = (grad_sum, (Range**2).sum(), H*W)

        if config.simulate:
            mn = mn.unsqueeze(1)
            mx = mx.unsqueeze(1)

        # mn *= 0.5
        # mx *= 0.5
        # new_mx = torch.max(-mn, mx)     # TODO hack
        # mn = -new_mx
        # mx = 2 * new_mx

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
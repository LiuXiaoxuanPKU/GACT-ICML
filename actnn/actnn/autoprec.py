import torch
import torch.nn as nn
import numpy as np
import random
from actnn.utils import uniform_sample, random_sample, exp_recorder
import actnn.cpp_extension.calc_precision as ext_calc_precision

import pickle
# Automatically compute the precision for each tensor


class AutoPrecision:
    def init_from_dims(self, dims):
        self.dims = torch.tensor(dims, dtype=torch.long)
        self.L = self.dims.shape[0]
        # Sensitivity for each tensor, tied within each group
        self.C = torch.ones(self.L)
        self.bits = torch.ones(self.L, dtype=torch.int32) * self.abits
        self.total_bits = self.abits * self.dims.sum()
        self.order = torch.randperm(self.L)

    def __init__(self, model, quantizer, bits, max_bits,
                 work_dir, adapt_interval, log_iter, sample_grad_ratio, sample_method,
                 momentum=0.99, warmup_iter=100, debug=False):
        self.model = model

        self.quantizer = quantizer
        self.debug = debug

        self.dims = None

        self.abits = bits
        self.max_bits = max_bits
        self.perm = []

        self.initialized = False

        # For maintaining batch_grad and detecting overly large quantization variance
        self.momentum = momentum
        self.warmpup_iter = warmup_iter
        self.beta1 = 1e-7
        self.batch_grad = 0
        self.grad_var = 0
        self.adapt_interval = adapt_interval
        self.sample_method = sample_method
        self.sample_grad_ratio = sample_grad_ratio

        self.iter = 0
        self.log_iter = log_iter
        self.work_dir = work_dir

    def iterate_wrapper(self, backprop):
        if self.dims is None:
            self.init_from_dims(self.quantizer.dims)
        self.iterate(backprop)
        self.quantizer.bwd_fns = []

    def check_dict_equal(self, d1, d2):
        assert(len(d1) == len(d2))
        for k in d1:
            if isinstance(d1[k], dict):
                self.check_dict_equal(d1[k], d2[k])
            elif isinstance(d1[k], torch.Tensor):
                assert(d1[k].equal(d2[k]))
            else:
                assert(d1[k] == d2[k])

    def iterate(self, backprop):
        def sample_grad():
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.ravel())
            return torch.cat(grad, 0)

        def setup_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # TODO det_grad is actually not necessary
        def get_grad():
            # TODO this is somewhat tricky...
            # TODO setstate & getstate won't work, why?
            setup_seed(self.iter)
            
            # get bn module
            def get_bn(model):
                bns = []
                for name, child in model.named_children():
                    if isinstance(child, nn.BatchNorm1d):
                        bns.append(child)
                    elif isinstance(child, nn.BatchNorm2d):
                        bns.append(child)
                    elif isinstance(child, nn.BatchNorm3d):
                        bns.append(child)
                    else:
                        ret = get_bn(child)
                        bns += ret
                return bns
            bn_layers = get_bn(self.model)
            for bn in bn_layers:
                bn.track_running_stats = False

            backprop()

            for bn in bn_layers:
                bn.track_running_stats = True
            grad = sample_grad()
            self.quantizer.iterate()

            return grad

        if (self.iter % self.adapt_interval == 0 and self.iter < 10 * self.adapt_interval) \
            or (self.iter % (10 * self.adapt_interval) == 0):
            # Do full adaptation
            print('ActNN: Initializing AutoPrec...')
            # different random seeds
            for l in range(self.L):
                b = self.quantizer.bits[l]
                self.quantizer.bits[l] = 1
                grad0 = get_grad()
                self.quantizer.seeds[l] += 1
                grad1 = get_grad()
                self.quantizer.seeds[l] -= 1
                self.quantizer.bits[l] = 8
                grad2 = get_grad()
                diff01 = ((grad0 - grad1) ** 2).sum()
                diff02 = ((grad0 - grad2) ** 2).sum()
                diff12 = ((grad1 - grad2) ** 2).sum()
                sens = max(diff01, max(diff02, diff12)*2)
                del grad0
                del grad1
                del grad2
                if torch.isnan(sens) or torch.isinf(sens) or sens > 1e10:
                    sens = 1e10
                self.C[l] = sens
                self.quantizer.bits[l] = b

            self.refresh_bits()

        if self.debug and self.iter % 10 == 0:
            # Debug information
            grad = get_grad()
            self.quantizer.bits = [32 for bit in self.bits]
            det_grad = get_grad()
            ap_var = ((grad - det_grad)**2).sum()
            predicted_var = (self.C * 2 ** (-2.0 * (self.bits - 1))).sum()
            print('ap var', ap_var.item(), predicted_var.item())

            for b in [4]:
                quant_var = 0
                for iter in range(1):
                    self.quantizer.bits = [b for bit in self.bits]
                    grad = get_grad()
                    var = ((grad - det_grad)**2).sum()
                    quant_var += var

                predicted_var = (self.C * 2 ** (-2.0 * (b - 1))).sum()
                print(b, ' bit ', quant_var.item(), predicted_var.item())
            self.quantizer.bits = [bit.item() for bit in self.bits]

        if self.log_iter > 0 and self.iter % self.log_iter == 0:
            det_grad = get_grad()

            # Maintain batch grad
            momentum = self.momentum
            self.beta1 = self.beta1 * momentum + 1 - momentum
            self.batch_grad = self.batch_grad * \
                momentum + (1 - momentum) * det_grad
            bgrad = self.batch_grad / self.beta1
            gvar = ((bgrad - det_grad)**2).sum()
            self.grad_var = self.grad_var * momentum + (1 - momentum) * gvar

            # log sensitivity information
            exp_recorder.record("iter", self.iter)
            exp_recorder.record("layer sensitivity", self.C.tolist())
            exp_recorder.record("bits", self.bits.tolist())
            exp_recorder.record("dims", self.dims.tolist())
            exp_recorder.record("grad_fn", self.quantizer.bwd_fns)
            exp_recorder.dump(self.work_dir + "autoprec.log")

        self.iter += 1

    def refresh_bits(self):
        total_bits = self.total_bits

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        self.bits = ext_calc_precision.calc_precision(self.bits,
                                                      self.C,
                                                      self.dims,
                                                      total_bits)

        self.quantizer.bits = [bit.item() for bit in self.bits]
        # Warning if the quantization variance is too large
        if self.log_iter > 0 and self.iter > self.warmpup_iter:
            overall_var = self.grad_var / self.beta1
            quantization_var = (
                self.C * 2 ** (-2 * self.bits.float())).sum().cuda()
            if quantization_var > overall_var * 0.1:
                print("========================================")
                print('ActNN Warning: Quantization variance is too large. Consider increasing number of bits.',
                      quantization_var, overall_var)
                exp_recorder.record("iter", self.iter)
                exp_recorder.record("layer sensitivity", self.C.tolist())
                exp_recorder.record("bits", self.bits.tolist())
                exp_recorder.record("dims", self.dims.tolist())
                exp_recorder.record("warning", True)
                exp_recorder.record("quantization var",
                                    quantization_var.tolist())
                exp_recorder.record("overall var", overall_var.tolist())
                exp_recorder.dump(self.work_dir + "autoprec.log")
                print("========================================")

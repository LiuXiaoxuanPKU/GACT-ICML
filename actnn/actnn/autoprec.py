import torch
import torch.nn as nn
import numpy as np
import random
from actnn.utils import uniform_sample, random_sample, exp_recorder
import actnn.cpp_extension.calc_precision as ext_calc_precision

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
                 work_dir, adapt_interval, sample_grad_ratio, sample_method,
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
        self.log_iter = 50
        self.work_dir = work_dir

        # self.refresh_bits()

    def iterate_wrapper(self, backprop):
        if self.dims is None:
            self.init_from_dims(self.quantizer.dims)
        self.iterate(backprop)

    def iterate(self, backprop):
        def sample_grad():
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.ravel())
                    # sample_cnt = max(min(10, param.grad.numel()),
                    #                  int(param.grad.numel() * self.sample_grad_ratio))
                    
                    # if self.sample_method == "random":
                    #     sample_grad = torch.tensor(random_sample(param.grad,
                    #                                             sample_cnt,
                    #                                             add_dataptr=False))
                    # elif self.sample_method == "uniform":
                    #     sample_grad = torch.tensor(uniform_sample(param.grad,
                    #                                             sample_cnt,
                    #                                             add_dataptr=False))
                    # else:
                    #     print("[Error] Unsupport sample method %s" % self.sample_method)
                    # grad.append(sample_grad)
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
            # The noise should be injected with other random seeds
            # TODO setstate & getstate won't work, why?
            # random.setstate(self.seeds[0])
            # np.random.set_state(self.seeds[1])
            # torch.set_rng_state(self.seeds[2])
            # torch.use_deterministic_algorithms(True)
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

            # random.setstate(self.seeds[0])
            # np.random.set_state(self.seeds[1])
            # torch.set_rng_state(self.seeds[2])

            return grad

        det_grad = None
        if self.iter % 2000 == 0:
            # Do full adaptation
            print('ActNN: Initializing AutoPrec...')
            # Option 1: different bits
            # self.quantizer.bits = [32 for bit in self.bits]
            # det_grad = get_grad()
            # for l in range(self.L):
            #     print("%d/%d" % (l, self.L))
            #     b = self.quantizer.bits[l]
            #     self.quantizer.bits[l] = 1
            #     grad = get_grad()
            #     self.quantizer.bits[l] = 32
            #     det_grad = get_grad()
            #     sens = ((grad - det_grad) ** 2).sum()
            #     del grad
            #     if torch.isnan(sens) or torch.isinf(sens) or sens > 1e10:
            #         sens = 1e10
            #     self.C[l] = sens
            #     self.quantizer.bits[l] = b

            if self.debug:
                grad = get_grad()
                self.quantizer.bits = [32 for bit in self.bits]
                det_grad = get_grad()
                ap_var = ((grad - det_grad) ** 2).sum()
                print('ap var', ap_var.item())
                self.quantizer.bits = [bit.item() for bit in self.bits]

            # Option 2: different random seeds
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
                # if self.debug:
                #     print("%d/%d" % (l, self.L), sens.item())

            # Option 3: inject noise
            # det_grad = get_grad()
            # for l in range(self.L):
            #     print("%d/%d" % (l, self.L))
            #     self.quantizer.inject_noises[l] = True
            #     grad = get_grad()
            #     sens = ((grad - det_grad) ** 2).sum() / 4
            #     del grad
            #     if torch.isnan(sens) or torch.isinf(sens) or sens > 1e10:
            #         sens = 1e10
            #     self.C[l] = sens
            #     self.quantizer.inject_noises[l] = False

            self.refresh_bits()

            # Output debug information
            if self.debug:
                # for l in range(self.L):
                #     print(l, 'Sensitivity: ', self.C[l].item(), ' bits: ', self.bits[l].item(), ' dims: ', self.dims[l])
                predicted_var = (self.C * 2 ** (-2.0 * (self.bits - 1))).sum()
                print('Predicted var ap ', predicted_var.item())

                grad = get_grad()
                self.quantizer.bits = [32 for bit in self.bits]
                det_grad = get_grad()
                ap_var = ((grad - det_grad)**2).sum()
                print('ap var', ap_var.item())
                #for b in [1, 2, 4, 8]:
                for b in [4]:
                    quant_var = 0
                    for iter in range(10):
                        self.quantizer.bits = [b for bit in self.bits]
                        for l in range(self.L):
                            self.quantizer.seeds[l] += 1
                        grad = get_grad()
                        var = ((grad - det_grad)**2).sum()
                        print(var)
                        quant_var += var

                    for l in range(self.L):
                        self.quantizer.seeds[l] = l

                    quant_var /= 10
                    predicted_var = (self.C * 2 ** (-2.0 * (b - 1))).sum()
                    print(b, ' bit ', quant_var.item(), predicted_var.item())

                self.quantizer.bits = [bit.item() for bit in self.bits]

        elif self.iter % self.adapt_interval == 0:
            if False: # Disable adaptation
                if len(self.perm) == 0:
                    self.perm = torch.randperm(self.L)
                l = self.perm[-1]
                self.perm = self.perm[:-1]

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
                sens = max(diff01, max(diff02, diff12) * 2)
                del grad0
                del grad1
                del grad2
                if torch.isnan(sens) or torch.isinf(sens) or sens > 1e10:
                    print('ActNN Warning: Large sensivitity ', sens.item())
                    sens = 1e10
                self.C[l] = sens
                self.quantizer.bits[l] = b

                self.refresh_bits()

            if self.debug:
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

        if self.iter % self.log_iter == 0:
            if det_grad is None:
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
        if self.iter > self.warmpup_iter:
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
                exp_recorder.dump("autoprec.log")
                print("========================================")

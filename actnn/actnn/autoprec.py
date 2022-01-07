import torch
import numpy as np
import random
from actnn.utils import uniform_sample, random_sample_perm, exp_recorder
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

    def __init__(self, model, quantizer, bits, max_bits, momentum=0.99,
                 adapt_interval=100, warmup_iter=100):
        self.model = model
        self.quantizer = quantizer

        self.dims = None

        self.abits = bits
        self.max_bits = max_bits
        self.perm = []

        self.initialized = False

        # For maintaining batch_grad and detecting overly large quantization variance
        self.momentum = momentum
        self.adapt_interval = 0
        self.warmpup_iter = warmup_iter
        self.beta1 = 1e-7
        self.batch_grad = 0
        self.grad_var = 0
        self.adapt_interval = adapt_interval

        self.iter = 0
        self.log_iter = 50

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
                    sample_cnt = max(min(10, param.grad.numel()),
                                     int(param.grad.numel() * 0.1))
                    sample_grad = torch.tensor(random_sample_perm(param.grad,
                                                                  sample_cnt,
                                                                  add_dataptr=False))
                    # sample_grad2 = torch.tensor(random_sample_perm(param.grad,
                    #                                                sample_cnt,
                    #                                                add_dataptr=False))
                    # print("sample cnt", sample_cnt)
                    # assert(sample_grad.equal(sample_grad2))
                    grad.append(sample_grad)
            return torch.cat(grad, 0)

        def setup_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

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
            backprop()
            grad = sample_grad()
            self.quantizer.iterate()

            # random.setstate(self.seeds[0])
            # np.random.set_state(self.seeds[1])
            # torch.set_rng_state(self.seeds[2])

            return grad

        det_grad = get_grad()
        # if self.iter == 0:
        #     org_bits = self.quantizer.bits
        #     self.quantizer.bits = [32 for _ in self.bits]
        #     det_grad = get_grad()
        #     for l in range(self.L):
        #         self.quantizer.bits[l] = 2
        #         grad = get_grad()
        #         print(l, ((det_grad - grad)**2).sum() * 4)
        #         del grad
        #         self.quantizer.bits[l] = 32

        if self.iter == 0:
            # Do full adaptation
            print('ActNN: Initializing AutoPrec...')
            # sum_c = 0
            for l in range(self.L):
                self.quantizer.inject_noises[l] = True
                grad = get_grad()
                self.C[l] = ((det_grad - grad) ** 2).sum() * 4
                self.quantizer.inject_noises[l] = False
            self.refresh_bits()

        elif self.iter % self.adapt_interval == 0:
            if len(self.perm) == 0:
                self.perm = torch.randperm(self.L)
            l = self.perm[-1]
            self.perm = self.perm[:-1]

            self.quantizer.inject_noises[l] = True
            grad = get_grad()
            self.C[l] = ((det_grad - grad) ** 2).sum() * \
                4  # Hack: always use 2bit
            self.quantizer.inject_noises[l] = False
            self.refresh_bits()

        # Maintain batch grad
        momentum = self.momentum
        self.beta1 = self.beta1 * momentum + 1 - momentum
        self.batch_grad = self.batch_grad * \
            momentum + (1 - momentum) * det_grad
        bgrad = self.batch_grad / self.beta1
        gvar = ((bgrad - det_grad)**2).sum()
        self.grad_var = self.grad_var * momentum + (1 - momentum) * gvar

        if self.iter % self.log_iter == 0:
            print("Iter: ", self.iter)
            print("[Layer Sensitivity]", self.C)
            print("[Bits]", self.bits)
            print("[Dims]", self.dims)
            exp_recorder.record("iter", self.iter)
            exp_recorder.record("layer sensitivity", self.C.tolist())
            exp_recorder.record("bits", self.bits.tolist())
            exp_recorder.record("dims", self.dims.tolist())
            exp_recorder.dump("autoprec.log")

        self.iter += 1

    def refresh_bits(self):
        total_bits = self.total_bits

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        self.bits = ext_calc_precision.calc_precision(self.bits,
                                                      self.C,
                                                      self.dims,
                                                      total_bits)

        self.quantizer.bits = [bit.item() for bit in self.bits]
        # print("Auto precision bits", self.bits)
        # Warning if the quantization variance is too large
        if self.iter > self.warmpup_iter:
            overall_var = self.grad_var / self.beta1
            quantization_var = (
                self.C * 2 ** (-2 * self.bits.float())).sum().cuda()
            if quantization_var > overall_var * 0.1:
                print("========================================")
                print('ActNN Warning: Quantization variance is too large. Consider increasing number of bits.',
                      quantization_var, overall_var)
                print("Iter: ", self.iter)
                print("[Layer Sensitivity]", self.C)
                print("[Bits]", self.bits)
                print("[Dims]", self.dims)
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

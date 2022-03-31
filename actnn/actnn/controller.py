import torch
from actnn.conf import config
from actnn.quantizer import Quantizer
from actnn.autoprec import AutoPrecision


class Controller:
    def __init__(self, model):
        if not config.compress_activation:
            return

        self.model = model

        if config.bit <= 4:
            default_bit = 4
        elif 4 < config.bit and config.bit <= 8:
            default_bit = 8
        else:
            assert(config.bit <= 16)
            default_bit = 8    
        
        self.quantizer = Quantizer(
            default_bit=default_bit, swap=config.swap, debug=config.debug, prefetch=config.prefetch)
        self.quantizer.filter_tensors(model.named_parameters())

        self.auto_prec = config.auto_prec
        if self.auto_prec:
            self.ap = AutoPrecision(
                self.model, self.quantizer, config.bit, config.max_bit, 
                config.work_dir, config.adapt_interval, config.log_interval)
        
        self.bit = config.bit
        self.iter = 0
        
    def __del__(self):
        self.uninstall_hook()

    def install_hook(self):
        def pack_hook(x):
            r = self.quantize(x)
            del x
            return r
        
        def unpack_hook(x):
            r = self.dequantize(x)
            del x
            return r
        torch._C._autograd._register_saved_tensors_default_hooks(
            pack_hook, unpack_hook)
    
    def uninstall_hook(self):
        torch._C._autograd._reset_saved_tensors_default_hooks()
        
    def iterate(self, get_grad):
        if not config.compress_activation:
            return

        self.quantizer.iterate()
        if self.auto_prec:
            self.ap.iterate_wrapper(get_grad)
        self.iter += 1
        self.quantizer.seed_iter = self.iter
        
    def quantize(self, input):
        if not config.compress_activation:
            if config.swap:
                # swap original tensor to cpu
                tensor_cpu = torch.empty(input.shape, dtype=input.dtype, device='cpu', pin_memory=True)
                tensor_cpu.copy_(input, non_blocking=True)
                return tensor_cpu
            else:
                return input
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        if not config.compress_activation:
            if config.swap:
                 input = input.cuda(non_blocking=True)
            return input
        return self.quantizer.dequantize(input)

from actnn.conf import config
from actnn.quantizer import Quantizer
from actnn.autoprec import AutoPrecision


class Controller:
    def __init__(self, model):
        if not config.compress_activation:
            return

        self.model = model

        if config.bit == 3:
            default_bit = 4
        elif config.bit > 4 and config.bit < 8:
            default_bit = 8
        else:
            default_bit = config.bit
        assert(8 % default_bit == 0)
        self.quantizer = Quantizer(
            default_bit=default_bit, swap=config.swap, debug=False, prefetch=config.prefetch)
        self.quantizer.filter_tensors(model.named_parameters())

        self.auto_prec = config.auto_prec
        if self.auto_prec:
            self.ap = AutoPrecision(
                self.model, self.quantizer, config.bit, config.max_bit, 
                config.work_dir, config.adapt_interval, config.log_interval,
                config.sample_grad_ratio, config.sample_method)
        self.bit = config.bit

        self.iter = 0

    def quantize(self, input):
        if not config.compress_activation:
            return input
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        if not config.compress_activation:
            return input
        return self.quantizer.dequantize(input)

    def iterate(self, get_grad):
        if not config.compress_activation:
            return

        self.quantizer.iterate()
        if self.auto_prec:
            self.ap.iterate_wrapper(get_grad)
        self.iter += 1

from actnn.conf import config
from actnn.quantizer import Quantizer
from actnn.autoprec import AutoPrecision


class Controller:
    def __init__(self, model):

        self.model = model

        self.quantizer = Quantizer(
            default_bit=config.bit, swap=config.swap, debug=False, prefetch=config.prefetch)
        self.quantizer.filter_tensors(model.named_parameters())

        self.auto_prec = config.auto_prec
        if self.auto_prec:
            self.ap = AutoPrecision(
                self.model, self.quantizer, config.bit, config.max_bit)
        self.bit = config.bit

        self.iter = 0

    def quantize(self, input):
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        return self.quantizer.dequantize(input)

    def iterate(self, get_grad):
        self.quantizer.iterate()
        if self.auto_prec:
            self.ap.iterate_wrapper(get_grad)
        self.iter += 1

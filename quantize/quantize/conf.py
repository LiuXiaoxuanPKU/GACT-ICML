class QuantizationConfig:
    def __init__(self):
        self.compress_activation = True
        self.activation_compression_bits = 2
        self.pergroup = True
        self.perlayer = True
        self.initial_bits = 8
        self.simulate = False
        self.stochastic = True
        self.swap = False
        self.training = True
        self.group_size = 256
        self.use_gradient = False

config = QuantizationConfig()

class QuantizationConfig:
    def __init__(self):
        self.compress_activation = False
        self.activation_compression_bits = 8
        self.persample = True
        self.perlayer = False
        self.initial_bits = 8
        self.simulate = True
        self.swap = False
        self.training = True
        self.group_dims = [0, 1, 2]
        self.qat = 0

config = QuantizationConfig()

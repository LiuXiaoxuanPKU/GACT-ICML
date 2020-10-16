class QuantizationConfig:
    def __init__(self):
        self.compress_activation = False
        self.activation_compression_bits = 8
        self.perlayer = False
        self.initial_bits = 8
        self.simulate = True
        self.swap = False

config = QuantizationConfig()

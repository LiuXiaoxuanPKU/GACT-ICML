import os

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

        # Memory management flag
        self.empty_cache = False

        # Debug related flag
        self.debug_memory_model = bool(os.environ.get('DEBUG_MEM_MODEL', False))
        self.debug_memory_op_forward = False
        self.debug_memory_op_backward = False
        self.debug_remove_bn = False
        self.debug_remove_relu = False

config = QuantizationConfig()

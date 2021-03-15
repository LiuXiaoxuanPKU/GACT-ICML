import ast
import os

class QuantizationConfig:
    def __init__(self):
        self.compress_activation = True
        self.activation_compression_bits = [2, 8, 8]
        self.pergroup = True
        self.perlayer = True
        self.initial_bits = 8
        self.stochastic = True
        self.training = True
        self.group_size = 256
        self.use_gradient = False
        self.adaptive_conv_scheme = True
        self.adaptive_bn_scheme = True
        self.simulate = False
        self.enable_quantized_bn = True

        # Memory management flag
        self.empty_cache_threshold = None
        self.pipeline_threshold = None
        self.cudnn_benchmark_conv2d = True
        self.swap = False

        # Debug related flag
        self.debug_memory_model = ast.literal_eval(os.environ.get('DEBUG_MEM', "False"))
        self.debug_speed = ast.literal_eval(os.environ.get('DEBUG_SPEED', "False"))
        self.debug_memory_op_forward = False
        self.debug_memory_op_backward = False
        self.debug_remove_bn = False
        self.debug_remove_relu = False

config = QuantizationConfig()

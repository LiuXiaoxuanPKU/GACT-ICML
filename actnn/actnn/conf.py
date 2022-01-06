import ast
import os


def set_optimization_level(level):
    if level == 'L0':      # Do nothing
        config.compress_activation = False
    elif level == 'L1':    # fixed 4-bit
        config.auto_prec = False
        config.bit = 4
    elif level == 'L1.1':    # fixed 8-bit
        config.auto_prec = False
        config.bit = 8
    elif level == 'L2':    # auto precision 4-bit
        config.auto_prec = True
        config.bit = 4
    elif level == 'L2.1':   # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
    elif level == 'L3':  # auto precision 4-bit + swap
        config.auto_prec = True
        config.bit = 4
        config.swap = True
    elif level == 'L3.1':  # auto precision + swap + prefetch
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
    elif level == 'L4':    # auto precision + swap + prefetch + defragmentation
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.defrag = True  # TODO: implement memory defragmentation
    elif level == 'swap':
        # TODO: implement naive swap
        config.swap = True
        config.compress_activation = False
    else:
        raise ValueError("Invalid level: " + level)


class QuantizationConfig:
    def __init__(self):
        self.compress_activation = True
        self.max_bit = 32
        self.bit = 4
        self.group_size = 256
        self.auto_prec = True

        # Memory management flag
        self.empty_cache_threshold = None
        self.swap = False
        self.prefetch = False
        self.defrag = False

        # Debug related flag
        self.debug_memory_model = ast.literal_eval(
            os.environ.get('DEBUG_MEM', "False"))
        self.debug_speed = ast.literal_eval(
            os.environ.get('DEBUG_SPEED', "False"))
        self.debug_memory_op_forward = False
        self.debug_memory_op_backward = False
        self.debug_remove_bn = False
        self.debug_remove_relu = False


config = QuantizationConfig()

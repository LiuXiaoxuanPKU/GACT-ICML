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
    elif level == 'L1.2':    # fixed 2-bit
        config.auto_prec = False
        config.bit = 2
    elif level == 'L2': # auto precision 4-bit
        config.auto_prec = True
        config.bit = 4
    elif level == 'L2.1': # auto precision 3-bit
        config.auto_prec = True
        config.bit = 3
    elif level == 'L2.2': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
    elif level == 'L3':  # auto precision 4-bit + swap
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
    elif level == 'swap': # naive swap
        config.swap = True
        config.compress_activation = False
    elif level == 'L4bit-swap':
        config.bit = 4
        config.swap = True
        config.auto_prec = False
    elif level == 'L4bit-swap-prefetch':
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.auto_prec = False
    else:
        raise ValueError("Invalid level: " + level)


class QuantizationConfig:
    def __init__(self):
        self.compress_activation = True
        self.max_bit = 32
        self.bit = 4
        self.group_size = 256
        self.auto_prec = True
        self.work_dir = "./log/" 
        self.adapt_interval = 100
        self.log_interval = 1000
        
        self.debug = False
        self.check_dup = True

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


config = QuantizationConfig()

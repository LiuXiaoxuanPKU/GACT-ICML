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
    elif level == 'L2.0': # auto precision 4-bit, do not check duplicate
        config.check_dup = False
        config.auto_prec = True
        config.bit = 4
    elif level == 'L2.4':    # auto precision 4-bit
        config.auto_prec = True
        config.bit = 4
    elif level == 'L2.3':   # auto precision 3-bit
        config.auto_prec = True
        config.bit = 3
    elif level == 'L2.2':   # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
        
    elif level == 'LH.1': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 3
        config.check_dup = False
        config.empty_cache_threshold = None
        
    elif level == 'LH.2': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 3
        config.check_dup = False
        config.empty_cache_threshold = 0.8
        
    elif level == 'LH.3': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 3
        config.check_dup = True
        config.empty_cache_threshold = None
        
    elif level == 'LH.4': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 3
        config.check_dup = True
        config.empty_cache_threshold =  0.8
        
    elif level == 'LF.1': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 4
        config.check_dup = False
        config.empty_cache_threshold = None
        
    elif level == 'LF.2': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 4
        config.check_dup = False
        config.empty_cache_threshold =  0.8
        
    elif level == 'LF.3': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 4
        config.check_dup = True
        config.empty_cache_threshold = None
        
    elif level == 'LF.4': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 4
        config.check_dup = True
        config.empty_cache_threshold = 0.8
    
    elif level == 'LT.1': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
        config.check_dup = False
        config.empty_cache_threshold = None
        
    elif level == 'LT.2': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
        config.check_dup = False
        config.empty_cache_threshold =  0.8
        
    elif level == 'LT.3': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
        config.check_dup = True
        config.empty_cache_threshold = None
        
    elif level == 'LT.4': # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
        config.check_dup = True
        config.empty_cache_threshold =  0.8
        
    elif level == 'L3':  # auto precision 4-bit + swap
        config.auto_prec = True
        config.bit = 4
        config.swap = True
    elif level == 'L3.1':  # auto precision + swap + prefetch + no_check_dup + no empty_cache
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.check_dup = False
        config.empty_cache_threshold = None
        
    elif level == 'L3.2':  # auto precision + swap + prefetch + check dup + no empty_cache
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.check_dup = True
        config.empty_cache_threshold = None
        
    elif level == 'L3.3':  # auto precision + swap + prefetch + no_check_dup + empty_cache
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.check_dup = False
        config.empty_cache_threshold = 0.7
        
    elif level == 'L3.4':  # auto precision + swap + prefetch + check dup + empty_cache
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.check_dup = True
        config.empty_cache_threshold = 0.7
    
    elif level == 'L4':    # auto precision + swap + prefetch + defragmentation
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.threshold = 1
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
        self.work_dir = "./log/" 
        self.adapt_interval = 20
        self.sample_grad_ratio = 0.1
        self.sample_method = 'uniform'
        
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

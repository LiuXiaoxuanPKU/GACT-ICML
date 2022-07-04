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
    elif level == 'L2':  # auto precision 4-bit
        config.auto_prec = True
        config.bit = 4
    elif level == 'L2.1':  # auto precision 3-bit
        config.auto_prec = True
        config.bit = 3
    elif level == 'L2.2':  # auto precision 2-bit
        config.auto_prec = True
        config.bit = 2
    elif level == 'L3':  # auto precision 4-bit + swap
        config.auto_prec = True
        config.bit = 4
        config.swap = True
        config.prefetch = True
    elif level == 'swap':  # vanilla swap
        config.swap = True
        config.compress_activation = False
    elif level == 'L4bit-swap': # fix 4 bit with swap
        config.bit = 4
        config.swap = True
        config.prefetch = True
        config.auto_prec = False
    else:
        raise ValueError("Invalid level: " + level)

def set_adapt_interval(i):
    config.adapt_interval = i
    
class QuantizationConfig:
    def __init__(self):
        # compress activation, this field is set to False when optimization level is L0
        self.compress_activation = True

        # ================== general quantization setting ================== 
        # average number of bits for activation
        # if auto precision is turned on, each activation is quantized uniformly with self.bit bits
        self.bit = 4
        self.group_size = 256
        # avoid the same activation multiple times, this will further reduce training memory
        # please reach out to xiaoxuan_liu@berkeley.edu if you meet bugs after setting this field to True
        self.check_dup = False

        # ================== auto precision ================== 
        self.auto_prec = True  # if auto precision is turned on
        # max number of bits for quantization, this field is only used for auto precision
        self.max_bit = 32
        self.adapt_interval = 1000  # the interval to adapt activation sensitivity
        self.work_dir = "./log/"  # log debug information under the self.work_dir directory
        # self.log_interval >= self.adapt_interval to avoid log repeat information
        # set log_interval = -1 to disable logging
        self.log_interval = -1 
        # debug sensitivity (compare estimate sensitivity with true sensitivity) for auto precision
        self.debug = False

        # Memory management flag
        self.swap = False
        self.prefetch = False


config = QuantizationConfig()

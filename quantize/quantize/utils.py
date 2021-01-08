import torch
import numpy as np

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated


def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if x.dtype in [torch.float32, torch.int]:
            ret += np.prod(x.size()) * 4 
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8]:
            ret += np.prod(x.size()) * 2

    return ret


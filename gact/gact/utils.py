from collections import OrderedDict
import json
import torch
import numpy as np

def uniform_sample_ref(input, sample_cnt, add_dataptr=True):
    step = max(torch.numel(input) // sample_cnt, 1)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())
    for i in range(min(sample_cnt, torch.numel(input))):
        idx = i * step
        key.append(input.view(-1)[idx].item())
    return key

def uniform_sample(input, sample_cnt, add_dataptr=True):
    num_elem = input.numel()
    sample_cnt = min(num_elem, sample_cnt)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())
    key += input.ravel()[torch.arange(0, sample_cnt).to(torch.long) *
                          (num_elem // sample_cnt)].tolist()
    return key

def random_sample(input, sample_cnt, add_dataptr=True):
    num_elem = input.numel()
    rng_state = torch.get_rng_state()
    seed = input.dim()
    torch.manual_seed(seed)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())

    key += input.view(-1)[torch.randint(0, num_elem, (sample_cnt,))].tolist()

    torch.set_rng_state(rng_state)
    return key

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
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
        if type(x)== int:
            ret += 4
        elif x.dtype in [torch.long]:
            ret += np.prod(x.size()) * 8
        elif x.dtype in [torch.float32, torch.int]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8, torch.uint8]:
            ret += np.prod(x.size()) * 1
        else:
            print("[Error] unsupport datatype ", x.dtype)
            exit(0)

    return ret

def empty_cache(ratio):
    if ratio is None:
        return
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if reserved > 0 and allocated / reserved < ratio:
        torch.cuda.empty_cache()

class GlobalExpRecorder:
    def __init__(self):
        self.val_dict = OrderedDict()

    def record(self, key, value, float_round=6):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(value, float_round)

        self.val_dict[key] = value

    def dump(self, filename):
        with open(filename, "a") as fout:
            fout.write(json.dumps(self.val_dict) + '\n')
        print("Save exp results to %s" % filename)

    def clear(self):
        pass


exp_recorder = GlobalExpRecorder()

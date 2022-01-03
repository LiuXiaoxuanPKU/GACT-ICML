import random
import torch
import numpy as np

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def error_rate(q_input, input):
    e = (((q_input - input)**2).sum() / (input**2).sum())
    print(e)
    return e


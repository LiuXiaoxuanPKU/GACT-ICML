# gact : Activation Compressed Training

## Install
- Requirements
```
torch>=1.9
```

- Build
```bash
cd gact
pip install -v -e .
```

## Usage
```python
from gact.controller import Controller # import gact controller
controller = Controller(default_bit=4, swap=False, prefetch=False)
model = .... # define your model here

def pack_hook(tensor): # quantize hook
    return controller.quantize(tensor)

def unpack_hook(tensor): # dequantize hook
    return controller.dequantize(tensor)

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook): # install hook
    # training logic
    for epoch in ...
      for iter in ....
        ......
        controller.iterate() # update the controller for each iteration
            
```

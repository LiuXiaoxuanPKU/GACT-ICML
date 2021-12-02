# ActNN : Activation Compressed Training

## Install
- Requirements
```
torch>=1.9
```

- Build
```bash
cd actnn
pip install -v -e .
```

## Usage
```from actnn.controller import Controller # import actnn controller
controller = Controller(default_bit=4, swap=False, debug=False, prefetch=False)
model = .... # define your model here
controller.filter_tensors(model.named_parameters()) # do not quantize parameters

def pack_hook(tensor): # quantize hook
    return debug_controller.quantize(tensor)

def unpack_hook(tensor): # dequantize hook
    return debug_controller.dequantize(tensor)

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook): # install hook
    # training logic
    for epoch in ...
      for iter in ....
      ......
      controller.iterate() # update the controller for each iteration
            
```

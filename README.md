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
model = .... # define your model here
controller = Controller(controller)

def pack_hook(tensor): # quantize hook
    return controller.quantize(tensor)

def unpack_hook(tensor): # dequantize hook
    return controller.dequantize(tensor)

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook): # install hook
    # training logic
    for epoch in ...
      for iter in ....
        ......
        def backprop():
            model.train() # make sure you are in the training mode
            output = model(input) # forward
            loss = calculate_loss()
            optimizer.zero_grad()
            loss.backward()

        controller.iterate(backprop) # pass forward/backward to controller
            
```

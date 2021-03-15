# ActNN : Activation Compressed Training

## Install
- Requirements
```
torch==1.7.1
torchvision==0.8.2
```

- Build
```bash
cd actnn
pip install -v -e .
```

## Usage
[mem_speed_benchmark/train.py](mem_speed_benchmark/train.py) is an example on using ActNN for models from torchvision.
Basically, there are three steps.

- Step1: Convert the model to use ActNN's layers.  
```python
import actnn
model = actnn.QModule(model)
```

- Step2: Configure the optimization level  
ActNN provides several optimization knobs.
See `train.py::set_optimization_level` to learn how to set them.

- (Optional) Step3: Change the data loader  
If you want to enable the per-sample strategy, you have to update the dataloader to return sample indices.
See `train_loader` in [mem_speed_benchmark/train.py](mem_speed_benchmark/train.py) for example.

## Image Classification
See [image_classification](image_classification/)

## Sementic Segmentation
Will be added later.

## Benchmark Memory Usage and Training Speed
See [mem_speed_benchmark](mem_speed_benchmark/)


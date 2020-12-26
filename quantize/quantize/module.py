import torch
import torch.nn as nn
import types
from quantize.conf import config


def QModule(model):
    def train(self, mode: bool = True):
        print('QModule.train')
        self.training = mode
        for module in self.children():
            module.train(mode)
        config.training = mode
        return self

    def eval(self):
        print('QModule.eval')
        config.training = False
        return self.train(False)

    # def convert_layers(model):
    #     for layer in model.children():
    #         if isinstance(layer, nn.Conv2d):
    #

    model.train = types.MethodType(train, model)
    model.eval = types.MethodType(eval, model)
    return model

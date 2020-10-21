from quantize import config, QScheme, QBNScheme
from .utils import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from matplotlib.colors import LogNorm
from quantize.hutchinson import trace_Hessian
from quantize.weights import get_precision
from copy import deepcopy


def get_var(model_and_loss, optimizer, val_loader, num_batches=20):
    num_samples = 3
    # print(QF.num_samples, QF.update_scale, QF.training)
    model_and_loss.train()
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    data_iter = enumerate(val_loader)
    inputs = []
    targets = []
    indices = []
    config.compress_activation = False
    QScheme.update_scale = True
    QBNScheme.update_scale = True

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name: layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad

    # First pass
    cnt = 0
    batch_grad = None
    for i, (input, target, index) in tqdm(data_iter):
        QScheme.batch = index
        QBNScheme.batch = index
        cnt += 1

        inputs.append(input.clone().cpu())
        targets.append(target.clone().cpu())
        indices.append(index.copy())
        mean_grad = bp(input, target)
        batch_grad = dict_add(batch_grad, mean_grad)

        exit(0)
        if cnt == num_batches:
            break

    num_batches = cnt
    batch_grad = dict_mul(batch_grad, 1.0 / num_batches)
    QScheme.update_scale = False
    QBNScheme.update_scale = False

    if config.perlayer:
        config.compress_activation = True
        QScheme.batch = indices[0]
        QBNScheme.batch = indices[0]
        grad = bp(inputs[0].cuda(), targets[0].cuda())
        QScheme.allocate_perlayer()
        # QBNScheme.allocate_perlayer()

    total_var = None
    total_error = None
    total_bias = None
    sample_var = None
    for i, input, target, index in tqdm(zip(range(num_batches), inputs, targets, indices)):
        input = input.cuda()
        target = target.cuda()
        QScheme.batch = index
        QBNScheme.batch = index
        config.compress_activation = False
        exact_grad = bp(input, target)
        sample_var = dict_add(sample_var, dict_sqr(dict_minus(exact_grad, batch_grad)))

        mean_grad = None
        second_momentum = None
        config.compress_activation = True
        for iter in range(num_samples):
            grad = bp(input, target)

            mean_grad = dict_add(mean_grad, grad)
            total_error = dict_add(total_error, dict_sqr(dict_minus(exact_grad, grad)))
            second_momentum = dict_add(second_momentum, dict_sqr(grad))

        mean_grad = dict_mul(mean_grad, 1.0 / num_samples)
        second_momentum = dict_mul(second_momentum, 1.0 / num_samples)

        grad_bias = dict_sqr(dict_minus(mean_grad, exact_grad))
        total_bias = dict_add(total_bias, grad_bias)

        grad_var = dict_minus(second_momentum, dict_sqr(mean_grad))
        total_var = dict_add(total_var, grad_var)

    total_error = dict_mul(total_error, 1.0 / (num_samples * num_batches))
    total_bias = dict_mul(total_bias, 1.0 / num_batches)
    total_var = dict_mul(total_var, 1.0 / num_batches)

    all_qg = 0
    all_b = 0
    all_s = 0
    for k in total_var:
        g = (batch_grad[k]**2).sum()
        sv = sample_var[k].sum()
        v = total_var[k].sum()
        b = total_bias[k].sum()
        e = total_error[k].sum()
        avg_v = v / total_var[k].numel()

        all_qg += v
        all_b += b
        all_s += sv
        print('{}, grad norm = {}, sample var = {}, bias = {}, var = {}, avg_var = {}, error = {}'.format(k, g, sv, b, v, avg_v, e))

    print('Overall Bias = {}, Var = {}, SampleVar = {}'.format(all_b, all_qg, all_s))


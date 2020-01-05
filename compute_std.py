import torch
import sys
from image_classification.utils import *

prefix = sys.argv[1]
num_workers = int(sys.argv[2])
num_samples = 10


def load_state(prefix):
    weights = torch.load(prefix + "_weight.grad")
    errors = torch.load(prefix + "_0_error.grad")
    for k in errors:
        errors[k] = [errors[k]]
    for i in range(1, num_workers):
        error = torch.load("{}_{}_error.grad".format(prefix, i))
        for k in error:
            errors[k].append(error[k])

    for k in errors:
        errors[k] = torch.cat(errors[k], 0)

    return weights, errors


def key(a):
    return [int(i) for i in a.split('_')[1:4]]


batch_gradient = torch.load(prefix + "/grad_mean.grad")
sample_grad_std = torch.load(prefix + "/grad_std.grad")
quan_grad_std = torch.load(prefix + "/grad_std_quan.grad")
exact_weight, exact_errors = load_state(prefix + "/exact")

weight, errors = load_state(prefix + "/sample_0")
d_weight = dict_minus(weight, exact_weight)
d_errors = dict_minus(errors, exact_errors)
bias_weight = d_weight
bias_errors = d_errors
var_weight = dict_sqr(d_weight)
var_errors = dict_sqr(d_errors)

for i in range(1, num_samples):
    print(i)
    weight, errors = load_state(prefix + "/sample_{}".format(i))
    d_weight = dict_minus(weight, exact_weight)
    d_errors = dict_minus(errors, exact_errors)
    bias_weight = dict_add(bias_weight, d_weight)
    bias_errors = dict_add(bias_errors, d_errors)
    var_weight = dict_add(var_weight, dict_sqr(d_weight))
    var_errors = dict_add(var_errors, dict_sqr(d_errors))

bias_weight = dict_mul(bias_weight, 1.0 / num_samples)
bias_errors = dict_mul(bias_errors, 1.0 / num_samples)
std_weight = dict_sqrt(dict_mul(var_weight, 1.0 / num_samples))
std_errors = dict_sqrt(dict_mul(var_errors, 1.0 / num_samples))

print("======== Weights =========")
weight_names = list(bias_weight.keys())
weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
weight_names = list(set(weight_names))
weight_names.sort(key=key)
for k in weight_names:
    grad_bias = bias_weight[k + '_grad']
    grad_std = std_weight[k + '_grad']
    grad_std_2 = sample_grad_std[k + '_grad']
    quan_std = quan_grad_std[k + '_grad']
    grad_mean = batch_gradient[k + '_grad']

    print('{}, batch grad mean={}, quant bias={}, quant std={}, sample std={}, overall std={}'.format(k, grad_mean.abs().mean(),
                                                                             grad_bias.abs().mean(),
                                                                             grad_std.abs().mean(),
                                                                             grad_std_2.abs().mean(),
                                                                             quan_std.abs().mean()))

print("======== Errors =========")
error_names = list(bias_errors.keys())
error_names.sort(key=key)
for k in error_names:
    print('{}, exact={}, bias={}, std={}, rel_std={}'.format(k, exact_errors[k].abs().mean(), bias_errors[k].abs().mean(), std_errors[k].mean(), std_errors[k].mean()/exact_errors[k].abs().mean()))

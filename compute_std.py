import torch
import sys

prefix = sys.argv[1]
num_workers = 8
num_samples = 3


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


def dict_add(x, y):
    return {k: x[k] + y[k] for k in x}

def dict_minus(x, y):
    return {k: x[k] - y[k] for k in x}

def dict_sqr(x):
    return {k: x[k]**2 for k in x}

def dict_sqrt(x):
    return {k: torch.sqrt(x[k]) for k in x}

def dict_mul(x, a):
    return {k: x[k]*a for k in x}


exact_weight, exact_errors = load_state(prefix + "/exact")

weight, errors = load_state(prefix + "/sample_0")
d_weight = dict_minus(weight, exact_weight)
d_errors = dict_minus(errors, exact_errors)
bias_weight = d_weight
bias_errors = d_errors
var_weight = dict_sqr(d_weight)
var_errors = dict_sqr(d_errors)

for i in range(1, num_samples):
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
weight_names.sort()
for k in weight_names:
    print('{}, bias={}, std={}'.format(k, bias_weight[k].abs().mean(), std_weight[k].mean()))

print("======== Errors =========")
error_names = list(bias_errors.keys())
error_names.sort()
for k in error_names:
    print('{}, bias={}, std={}'.format(k, bias_errors[k].abs().mean(), std_errors[k].mean()))

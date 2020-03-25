import torch
from .quantize import config
from .utils import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle


def get_error_grad(m):
    grad_dict = {}

    if hasattr(m, 'layer4'):
        layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        layers = [m.layer1, m.layer2, m.layer3]

    for lid, layer in enumerate(layers):
        for bid, block in enumerate(layer):
            clayers = [block.conv1_in, block.conv2_in]
            if hasattr(block, 'conv3'):
                clayers.extend([block.conv3_in])

            for cid, clayer in enumerate(clayers):
                layer_name = 'conv_{}_{}_{}_error'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.grad.detach().cpu()

    return grad_dict


def get_grad(m):
    grad_dict = {}

    if hasattr(m, 'layer4'):
        layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        layers = [m.layer1, m.layer2, m.layer3]

    for lid, layer in enumerate(layers):
        for bid, block in enumerate(layer):
            clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                else [block.conv1, block.conv2]

            for cid, clayer in enumerate(clayers):
                layer_name = 'conv_{}_{}_{}_weight'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.detach().cpu()
                layer_name = 'conv_{}_{}_{}_grad'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.grad.detach().cpu()

    return grad_dict


def get_batch_grad(model_and_loss, optimizer, val_loader, ckpt_name):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    optimizer.zero_grad()
    cnt = 0
    for i, (input, target) in data_iter:
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        cnt += 1

    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.grad /= cnt

    grad = get_grad(m)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(grad, ckpt_name)
    return get_grad(m)


def get_grad_bias_std(model_and_loss, optimizer, val_loader, mean_grad, ckpt_name, num_epochs=1):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    empirical_mean_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss, output = model_and_loss(input, target)
            loss.backward()
            torch.cuda.synchronize()

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                grad_dict = get_grad(m)

                e_grad = dict_sqr(dict_minus(grad_dict, mean_grad))
                if var_grad is None:
                    var_grad = e_grad
                else:
                    var_grad = dict_add(var_grad, e_grad)

                if empirical_mean_grad is None:
                    empirical_mean_grad = grad_dict
                else:
                    empirical_mean_grad = dict_add(empirical_mean_grad, grad_dict)

            cnt += 1

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        std_grad = dict_sqrt(dict_mul(var_grad, 1.0 / cnt))
        bias_grad = dict_minus(dict_mul(empirical_mean_grad, 1.0/cnt), mean_grad)
        torch.save(std_grad, ckpt_name)
        return bias_grad, std_grad


def debug_bias(model_and_loss, optimizer, val_loader):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    empirical_mean_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        break

    config.quantize_gradient = False
    optimizer.zero_grad()
    loss, output = model_and_loss(input, target)
    loss.backward()
    torch.cuda.synchronize()

    exact_grad = get_grad(m)
    empirical_mean_grad = None
    config.quantize_gradient = True
    for e in range(100):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()

        cnt += 1
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            grad_dict = get_grad(m)

            if empirical_mean_grad is None:
                empirical_mean_grad = grad_dict
            else:
                empirical_mean_grad = dict_add(empirical_mean_grad, grad_dict)

            bias_grad = dict_minus(dict_mul(empirical_mean_grad, 1.0/cnt), exact_grad)
            print(e, bias_grad['conv_1_1_1_grad'].abs().mean())


def get_gradient(model_and_loss, optimizer, input, target, prefix):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        grad_dict = get_grad(m)
        ckpt_name = "{}_weight.grad".format(prefix)
        torch.save(grad_dict, ckpt_name)

    grad_dict = get_error_grad(m)
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()

    ckpt_name = "{}_{}_error.grad".format(prefix, rank)
    torch.save(grad_dict, ckpt_name)


def dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    config.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    # print("Computing gradient std...")
    # get_grad_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad")

    print("Computing quantization noise...")
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/exact")

    config.quantize_gradient = True
    for i in range(10):
        print(i)
        get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/sample_{}".format(i))

    # print("Computing quantized gradient std...")
    # get_grad_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad")


def key(a):
    return [int(i) for i in a.split('_')[1:4]]


def fast_dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    # debug_bias(model_and_loss, optimizer, val_loader)
    # exit(0)

    config.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    print("Computing gradient std...")
    g_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad", num_epochs=1)

    config.quantize_gradient = True
    q_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad", num_epochs=1)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        bias_grad, std_grad = g_outputs
        bias_quan, std_quan = q_outputs
        weight_names = list(grad.keys())
        weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
        weight_names = list(set(weight_names))
        weight_names.sort(key=key)
        for k in weight_names:
            grad_mean = grad[k + '_grad']
            sg = std_grad[k + '_grad']
            bg = bias_grad[k + '_grad']
            sq = std_quan[k + '_grad']
            bq = bias_quan[k + '_grad']

            print('{}, batch grad mean={}, sample std={}, sample bias={}, overall std={}, overall bias={}'.format(
                k, grad_mean.abs().mean(), sg.mean(), bg.abs().mean(), sq.mean(), bq.abs().mean()))


def fast_dump_2(model_and_loss, optimizer, val_loader, checkpoint_dir):
    config.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    print("Computing gradient std...")
    g_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad", num_epochs=1)

    config.quantize_gradient = True
    q_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad", num_epochs=1)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        bias_grad, std_grad = g_outputs
        bias_quan, std_quan = q_outputs
        weight_names = list(grad.keys())
        weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
        weight_names = list(set(weight_names))
        weight_names.sort(key=key)

        sample_var = 0.0
        overall_var = 0.0
        for k in weight_names:
            grad_mean = grad[k + '_grad']
            sg = std_grad[k + '_grad']
            sq = std_quan[k + '_grad']

            print('{}, batch grad norm={}, sample var={}, quantization var={}, overall var={}'.format(
                k, grad_mean.norm()**2, sg.norm()**2, sq.norm()**2-sg.norm()**2, sq.norm()**2))

            sample_var += sg.norm()**2
            overall_var += sq.norm()**2

        print('SampleVar = {}, QuantVar = {}, OverallVar = {}'.format(
            sample_var, overall_var - sample_var, overall_var))



def plot_bin_hist(model_and_loss, optimizer, val_loader):
    config.grads = []
    config.acts = []
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        num_grads = len(config.grads)
        fig, ax = plt.subplots(num_grads, figsize=(5, 5*num_grads))
        for i in range(num_grads):
            g, min_val, max_val = config.grads[i]
            ax[i].hist(g.ravel(), bins=2**config.backward_num_bits)
            ax[i].set_title(str(i))
            print(i, g.shape, min_val, max_val)

        fig.savefig('grad_hist.pdf')

        np.savez('errors.pkl', *config.grads)
        np.savez('acts.pkl', *config.acts)


def write_errors(model_and_loss, optimizer, val_loader):
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    for iter in range(10):
        print(iter)
        config.grads = []
        loss, output = model_and_loss(input, target)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            np.savez('errors_{}.pkl'.format(iter), *config.grads)

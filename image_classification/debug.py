import torch
import image_classification.quantize
from .utils import *


def get_batch_grad(model_and_loss, optimizer, val_loader):
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

    grad_dict = {}
    for lid, layer in enumerate([m.layer1, m.layer2, m.layer3, m.layer4]):
        for bid, block in enumerate(layer):
            for cid, clayer in enumerate([block.conv1, block.conv2, block.conv3]):
                layer_name = 'conv_{}_{}_{}_weight'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.detach().cpu()
                layer_name = 'conv_{}_{}_{}_grad'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.grad.detach().cpu()

    return grad_dict


def get_grad_std(model_and_loss, optimizer, val_loader, mean_grad, ckpt_name):
    data_iter = enumerate(val_loader)
    var_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()

        if torch.distributed.get_rank() == 0:
            grad_dict = {}
            for lid, layer in enumerate([m.layer1, m.layer2, m.layer3, m.layer4]):
                for bid, block in enumerate(layer):
                    for cid, clayer in enumerate([block.conv1, block.conv2, block.conv3]):
                        layer_name = 'conv_{}_{}_{}_weight'.format(lid + 1, bid + 1, cid + 1)
                        grad_dict[layer_name] = clayer.weight.detach().cpu()
                        layer_name = 'conv_{}_{}_{}_grad'.format(lid + 1, bid + 1, cid + 1)
                        grad_dict[layer_name] = clayer.weight.grad.detach().cpu()

            e_grad = dict_sqr(dict_minus(grad_dict - mean_grad))
            if var_grad is None:
                var_grad = e_grad
            else:
                var_grad = dict_add(var_grad, e_grad)

        cnt += 1

    if torch.distributed.get_rank() == 0:
        var_grad = dict_sqrt(dict_mul(var_grad, 1.0 / cnt))
        torch.save(grad_dict, ckpt_name)


def get_gradient(model_and_loss, optimizer, input, target, prefix):
    m = model_and_loss.model.module
    m.set_debug(True)

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    if torch.distributed.get_rank() == 0:
        grad_dict = {}
        for lid, layer in enumerate([m.layer1, m.layer2, m.layer3, m.layer4]):
            for bid, block in enumerate(layer):
                for cid, clayer in enumerate([block.conv1, block.conv2, block.conv3]):
                    layer_name = 'conv_{}_{}_{}_weight'.format(lid + 1, bid + 1, cid + 1)
                    grad_dict[layer_name] = clayer.weight.detach().cpu()
                    layer_name = 'conv_{}_{}_{}_grad'.format(lid+1, bid+1, cid+1)
                    grad_dict[layer_name] = clayer.weight.grad.detach().cpu()

        ckpt_name = "{}_weight.grad".format(prefix)
        torch.save(grad_dict, ckpt_name)

    grad_dict = {}
    for lid, layer in enumerate([m.layer1, m.layer2, m.layer3, m.layer4]):
        for bid, block in enumerate(layer):
            for cid, clayer in enumerate([block.conv1_in, block.conv2_in, block.conv3_in]):
                layer_name = 'conv_{}_{}_{}_error'.format(lid+1, bid+1, cid+1)
                grad_dict[layer_name] = clayer.grad.detach().cpu()

    ckpt_name = "{}_{}_error.grad".format(prefix, torch.distributed.get_rank())
    torch.save(grad_dict, ckpt_name)


def dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader)

    print("Computing gradient std...")
    get_grad_std(model_and_loss, optimizer, grad, checkpoint_dir + "/grad_std.grad")

    print("Computing quantization noise...")
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:8]
    target = target[:8]

    image_classification.quantize.quan_grad = False
    get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/exact")

    image_classification.quantize.quan_grad = True
    for i in range(3):
        print(i)
        get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/sample_{}".format(i))

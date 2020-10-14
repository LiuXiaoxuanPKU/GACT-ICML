import torch
from torch.nn import Conv2d, BatchNorm2d, Linear
from quantize import config, QScheme, QConv2d, QBatchNorm2d, QLinear

config.compress_activation = True
config.activation_compression_bits = 2
QScheme.update_scale = False
QScheme.num_samples = 128
QScheme.batch = torch.arange(0, 128)
num_samples = 1000


def rel_error(a, b):
    return (a - b).norm() / a.norm()


def test_layer(layer, qlayer, get_params, input):
    output = layer(input)
    qoutput = qlayer(input)
    print('Output error ', rel_error(output, qoutput))

    weight = torch.randn_like(output)
    loss = (output * weight).sum()
    loss.backward()
    grad = get_params(layer).grad.detach().clone()
    # print(grad[0,0,0])

    qgrads = []
    for i in range(num_samples):
        qoutput = qlayer(input)
        qloss = (qoutput * weight).sum()

        get_params(qlayer).grad = None
        qloss.backward()
        qgrads.append(get_params(qlayer).grad.detach().clone())

    qgrads = torch.stack(qgrads, 0)
    qgrad_mean = qgrads.mean(0)
    qgrad_std = qgrads.std(0)
    # print(qgrad_mean[0, 0, 0])
    # for i in range(10):
    #     print(qgrads[i, 0, 0, 0])
    #
    # print((grad - qgrad_mean)[0,0,0])
    # print(qgrad_std[0,0,0])

    bias = ((grad - qgrad_mean) ** 2).sum()
    var = (qgrad_std ** 2).sum()

    print('Norm = {}, bias = {}, var = {}'.format((grad**2).sum(), bias, var))


# print('================')
# model = Conv2d(16, 32, 3, bias=False).cuda()
# qmodel = QConv2d(16, 32, 3, bias=False).cuda()
# qmodel.weight = model.weight
# test_layer(model, qmodel,
#            lambda model: model.weight, torch.randn((128,16,32,32)).cuda())


# print('================')
# model = Linear(100, 200, False).cuda()
# qmodel = QLinear(100, 200, False).cuda()
# qmodel.weight = model.weight
# test_layer(model, qmodel,
#            lambda model: model.weight, torch.randn((128, 100)).cuda())

print('================')
model = BatchNorm2d(16).cuda()
qmodel = QBatchNorm2d(16).cuda()
# model.eval()
# qmodel.eval()
with torch.no_grad():
    qmodel.weight.copy_(model.weight)
    qmodel.bias.copy_(model.bias)

input = torch.randn((128, 16, 32, 32)).cuda()
# input.requires_grad_()

test_layer(model, qmodel,
           lambda model: model.weight, input)
import torch
torch.manual_seed(0)
import torch.nn as nn
from tqdm import tqdm
import time

# trace(A) = E[x^\top A x]        x ~ N(0, I)

def hutchinson(A, num_samples=100):
    N = A.shape[0]
    x = torch.randn(N, num_samples)
    Ax = A @ x      # [N, samples]
    xAx = (Ax * x).sum(0)
    return xAx.mean()


# N = 10
# H = torch.randn(N, N)
# H = H + H.t()
#
# print('Real trace = ', H.diag().sum())
# for num_samples in [1, 10, 100, 1000, 10000]:
#     print(hutchinson(H, num_samples))


# trace(H) = E[x^\top H x] = E[x * (grad * x)']
def trace_Hessian(model_and_loss, params, inputs, num_samples=100):
    model = model_and_loss.model
    param_names = list(params.keys())
    param_vals = [params[k] for k in param_names]

    loss, _ = model_and_loss(*inputs)
    grad = torch.autograd.grad(loss, param_vals, create_graph=True)
    grad = {param_names[k]: grad[k] for k in range(len(params))}

    trace = {k: 0.0 for k in params}
    for iter in tqdm(range(num_samples)):
        aux_loss = 0.0
        x = {}
        for k in params:
            x[k] = torch.randn_like(params[k])
            aux_loss += (x[k] * grad[k]).sum()

        second_grad = torch.autograd.grad(aux_loss, param_vals, retain_graph=True)

        for idx in range(len(param_vals)):
            k = param_names[idx]
            trace[k] += (x[k] * second_grad[idx]).sum()

    return {k: trace[k] / num_samples for k in trace}


def trace_Hessian_exact(model_and_loss, params, inputs):
    model = model_and_loss.model
    param_names = list(params.keys())
    param_vals = [params[k] for k in param_names]

    loss = model_and_loss(*inputs)
    grad = torch.autograd.grad(loss, param_vals, create_graph=True)
    grad = {param_names[k]: grad[k] for k in range(len(params))}

    trace = {k: 0.0 for k in params}
    for k in params:
        for idx in range(params[k].shape[0]):
            second_grad = torch.autograd.grad(grad[k][idx], params[k], retain_graph=True)[0]
            trace[k] += second_grad[idx]

    return trace


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        A = torch.randn(3, 3)
        self.A = A @ A.t() * 0.01
        self.a = nn.Parameter(torch.randn(3), requires_grad=True)

        B = torch.randn(5, 5)
        self.B = B @ B.t()
        self.b = nn.Parameter(torch.randn(5), requires_grad=True)

        self.model = self

    def forward(self):
        return 0.5 * (((self.A @ self.a) * self.a).sum() + ((self.B @ self.b) * self.b).sum())


if __name__ == '__main__':
    model = Model()
    params = {'A': model.a, 'B': model.b}
    a = torch.randn(3)
    b = torch.randn(5)

    print('Real trace = ', model.A.diag().sum(), model.B.diag().sum())
    print(trace_Hessian(model, params, (), 30000))
    # print(trace_Hessian_exact(model, params, ()))

    # from torch import Tensor
    # from torch.autograd import Variable
    # from torch.autograd import grad
    # from torch import nn
    #
    # # some toy data
    # x = Variable(Tensor([4., 2.]), requires_grad=False)
    # y = Variable(Tensor([1.]), requires_grad=False)
    #
    # # linear model and squared difference loss
    # model = nn.Linear(2, 1)
    # loss = torch.sum((y - model(x)) ** 2)
    #
    # # instead of using loss.backward(), use torch.autograd.grad() to compute gradients
    # loss_grads = grad(loss, model.parameters(), create_graph=True)
    # print(loss_grads)
    #
    # # compute the second order derivative w.r.t. each parameter
    # d2loss = []
    # for param, grd in zip(model.parameters(), loss_grads):
    #     drv = grad(grd[idx], param[idx], create_graph=True)
    #     d2loss.append(drv)
    #     print(param, drv)

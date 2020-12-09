import torch

class LipschitzEstimator:
"""
Estimate the Lipschitz constant of the gradient
"""
    instances = []

    def __init__(self, weight):
        weight = weight.view(-1)
        N = weight.shape[0]
        self.indices = torch.randint(0, N-1, max(100, N//100))
        self.weight = weight
        self.lip = 1.0
        self.data = None
        self.grad = None
        LipschitzEstimator.instances.append(self)

    def update(self):
        last_data = self.data
        new_data = self.weight[self.indices].detach().clone()

        last_grad = self.grad
        new_grad = self.weight.grad[self.indices].detach().clone()
        if last_data is not None:
            self.lip = (new_grad - last_grad).norm() / (new_data - last_data).norm()

        self.data = new_data
        self.grad = new_grad

    @staticmethod
    def update_all():
        for estimator in instances:
            estimator.update()


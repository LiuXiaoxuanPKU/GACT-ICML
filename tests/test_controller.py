import random
import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0)

from actnn import get_memory_usage, compute_tensor_bytes
from actnn.controller import Controller

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TestNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=False),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(40000, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        print("pool output shape: ", x.shape)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


N = 128
x = torch.rand((N, 3, 224, 224), device="cuda")
num_classes = 100
y = torch.randint(0, num_classes, (N, num_classes), device="cuda").float()


def train(controller = None):
    set_random_seed(0)
    model = TestNet(num_classes=100).cuda()
    if controller is not None:
        controller.filter_tensors(model.named_parameters())
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    print("===========Start Train==========")
    losses = []
    mems = []
    for t in range(2):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if t % 10 == 9:
            losses.append(loss.item())
            print(t, loss.item())

        before_backward_mem = get_memory_usage()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after_backward_mem = get_memory_usage()
        mems.append(before_backward_mem - after_backward_mem)
        if controller:
            controller.iterate(model)
        print("----------------------------------")
    
    return losses, mems


if __name__ == "__main__":
    # original 
    orig_losses, org_mems = train()

    # 4 bit quantization
    controller = Controller(auto_prec=False, verbose=False, swap=False, single_quantize=True)
    def pack_hook(tensor):
        return controller.quantize(tensor)
    def unpack_hook(tensor):
        return controller.dequantize(tensor)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        q4_losses, q4_mems = train(controller)

    # 4 bit with swap
    swap_controller = Controller(auto_prec=False, verbose=False, swap=True, single_quantize=True)
    def pack_hook(tensor):
        return swap_controller.quantize(tensor)
    def unpack_hook(tensor):
        return swap_controller.dequantize(tensor)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        q4_swap_losses, q4_swap_mems = train(swap_controller)

    # test swap correctness
    np.testing.assert_allclose(q4_losses, q4_swap_losses)

    # test memory
    print("original activation mems %d MB" % (org_mems[-1]/1024/1024))
    print("4 bit activation mems %d MB" % (q4_mems[-1]/1024/1024))
    print("4 bit + swap activation mems %d MB" % (q4_swap_mems[-1]/1024/1024))



    

    

from gact.controller import Controller
from gact import get_memory_usage, compute_tensor_bytes
import time
import random
import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0)


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
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            # nn.Linear(46656, num_classes)
            nn.Linear(9216, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print("pool output shape: ", x.shape)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


N = 256
x = torch.rand((N, 3, 224, 224), device="cuda")
num_classes = 100
y = torch.randint(0, num_classes, (N, num_classes), device="cuda").float()


def train(controller=None):
    set_random_seed(0)
    model = TestNet(num_classes=100).cuda()
    if controller is not None:
        controller.filter_tensors(model.named_parameters())
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    print("===========Start Train==========")
    losses = []
    mems = []
    train_ips_list = []
    model.train()
    end = time.time()
    for t in range(20):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if t % 1 == 0:
            losses.append(loss.item())
            print(t, loss.item())

        before_backward_mem = get_memory_usage()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after_backward_mem = get_memory_usage()
        mems.append(before_backward_mem - after_backward_mem)
        bs = x.shape[0]
        batch_total_time = time.time() - end
        train_ips = bs / batch_total_time
        train_ips_list.append(train_ips)
        end = time.time()
        if controller:
            controller.iterate()
        # print("----------------------------------")

    return losses, mems, train_ips_list


if __name__ == "__main__":
    # original warmup
    orig_losses, org_mems, org_ips_list = train()

    # original
    orig_losses, org_mems, org_ips_list = train()

    # 4 bit debug
    debug_controller = Controller(verbose=False, swap=False)

    def pack_hook(tensor):
        return debug_controller.quantize(tensor)

    def unpack_hook(tensor):
        return debug_controller.dequantize(tensor)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        q4_debug_losses, q4_debug_mems, q4_debug_ips_list = train(
            debug_controller)

    # 4 bit quantization
    controller = Controller(verbose=False, swap=False)

    def pack_hook(tensor):
        return controller.quantize(tensor)

    def unpack_hook(tensor):
        return controller.dequantize(tensor)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        q4_losses, q4_mems, q4_ips_list = train(controller)

    # 4 bit with swap
    swap_controller = Controller(
        verbose=False, swap=True, prefetch=True)

    def pack_hook(tensor):
        return swap_controller.quantize(tensor)

    def unpack_hook(tensor):
        return swap_controller.dequantize(tensor)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        q4_swap_losses, q4_swap_mems, q4_swap_ips_list = train(swap_controller)

    # test swap correctness
    np.testing.assert_allclose(q4_debug_losses, q4_losses)
    np.testing.assert_allclose(q4_losses, q4_swap_losses)

    # test memory
    print("=========Test Memory============")
    print("original activation mems %d MB" % (org_mems[-1]/1024/1024))
    print("4 bit debug mems %d MB" % (q4_debug_mems[-1]/1024/1024))
    print("4 bit activation mems %d MB" % (q4_mems[-1]/1024/1024))
    print("4 bit + swap activation mems %d MB" % (q4_swap_mems[-1]/1024/1024))

    # test speed
    print("=========Test Speed============")
    print("original speed %f ips" % (np.median(org_ips_list)))
    print("4 bit debug speed %f ips" % (np.median(q4_debug_ips_list)))
    print("4 bit speed %f ips" % (np.median(q4_ips_list)))
    print("4 bit + swap speed %f ips" % (np.median(q4_swap_ips_list)))

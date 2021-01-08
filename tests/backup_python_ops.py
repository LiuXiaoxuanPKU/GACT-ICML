"""Backup deprecated python op implementation"""

class quantized_max_pool2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
        assert not ceil_mode and not return_indices
        assert dilation[0] == dilation[1] == 1
        assert kernel_size[0] * kernel_size[1] < 2 ** 4
        N = input.shape[0]
        C = input.shape[1]
        H = input.shape[2]
        W = input.shape[3]
        H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        out = torch.empty((N, C, H_out, W_out), device=input.device)
        max_index = torch.empty((N, C, H_out, W_out), dtype=torch.int8, device=input.device)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_base = h * stride[0] - padding[0]
                        h_start = max(h_base, 0)
                        h_end = min(h_base + kernel_size[0], H)
                        w_base = w * stride[1] - padding[1]
                        w_start = max(w_base, 0)
                        w_end = min(w_base + kernel_size[1], W)

                        v = -1e10
                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                if input[n, c, i, j] > v:
                                    v = input[n, c, i, j]
                                    index = (i - h_base) * kernel_size[1] + (j - w_base)
                                    assert 0 <= index < kernel_size[0] * kernel_size[1]

                        out[n, c, h, w] = v
                        max_index[n, c, h, w] = index
        ctx.saved = (input.shape, max_index, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, max_index, kernel_size, stride, padding, dilation, ceil_mode, return_indices = ctx.saved
        grad_input = torch.zeros(input_shape, device=grad_output.device)
        N, C, H_out, W_out = grad_output.shape
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_base = h * stride[0] - padding[0]
                        w_base = w * stride[1] - padding[1]
                        index = max_index[n, c, h, w]
                        h_offset, w_offset = index // kernel_size[1], index % kernel_size[1]
                        grad_input[n, c, h_base + h_offset, w_base + w_offset] += grad_output[n, c, h, w]

        return grad_input, None, None, None, None, None, None


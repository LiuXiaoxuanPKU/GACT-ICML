import torch
from gact.ops import op_quantize, op_dequantize
from timeit_v2 import py_benchmark

def test_speed():
    shape = (256, 256, 14, 14)
    data = torch.rand(shape).cuda()
    stmt = "op_quantize(data, 2)"
    t_quantize = py_benchmark(stmt, {**globals(), **locals()},
                                 setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
    output = op_quantize(data, 2)
    stmt = "op_dequantize(output, shape, False)"
    t_dequantize = py_benchmark(stmt, {**globals(), **locals()},
                                 setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
    print("quantize %f, dequantize %f" % (t_quantize, t_dequantize))
    
if __name__ == "__main__":
    test_speed()
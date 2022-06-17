import torch
import gact
from gact.utils import uniform_sample_ref, uniform_sample, random_sample
from timeit_v2 import py_benchmark
from utils import setup_seed, error_rate
import time


def test_sample_correctness():
    print("======== Test Sample Correctness ==========")
    data = torch.randn((800, 512, 768)).cuda()

    def test_implementation(sample_method):
        sample1 = tuple(sample_method(data, 10))
        sample2 = tuple(sample_method(data, 10))
        assert(len(sample1) == 11)
        assert(sample1 == sample2)
        return sample1

    s1 = test_implementation(uniform_sample)
    s2 = test_implementation(uniform_sample_ref)
    assert(tuple(s1) == tuple(s2))
    s3 = test_implementation(random_sample)


def test_sample_speed():
    print("======== Test Sample Speed ==========")
    data = torch.randn((800, 512, 768)).cuda()

    def test_ref_nonzero_mem():
        # relu = torch.nn.ReLU(inplace=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        print("Start Mem:", start_mem / 1024 / 1024)
        cnt = data.count_nonzero()
        # cnt = relu(data)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        # print(cnt)
        return (peak_mem - start_mem) / 1024 / 1024
    
    def test_ref_nonzero_speed():
        torch.cuda.synchronize()
        start = time.time()
        cnt = data.count_nonzero()
        torch.cuda.synchronize()
        end = time.time()
        print(cnt)
        return end - start

    def test_ref_relu_speed():
        relu = torch.nn.ReLU()
        torch.cuda.synchronize()
        start = time.time()
        ret = relu(data)
        torch.cuda.synchronize()
        end = time.time()
        print(ret.ravel()[0])
        return end - start

    def test_implementation(sample_method, sample_cnt=100):
        torch.cuda.synchronize()
        start = time.time()
        ret = sample_method(data, sample_cnt)
        torch.cuda.synchronize()
        end = time.time()
        print(ret[0])
        return end - start

    mem_count_zero = test_ref_nonzero_mem()
    t_count_zero = test_ref_nonzero_speed()
    t_relu = test_ref_relu_speed()
    t_uniform_sample = test_implementation(uniform_sample)
    t_random_sample_perm_100 = test_implementation(random_sample, 100)
    t_random_sample_perm_30 = test_implementation(random_sample, 1000)
    print("Nonzero speed %.10f ms, mem %.10f MB" % (t_count_zero, mem_count_zero))
    print("ReLU speed %.10f ms" % t_relu)
    print("Uniform sample speed %.10f ms" % t_uniform_sample)
    print("Random sample speed 1 %.10f ms" % t_random_sample_perm_100)
    print("Random sample speed 2 %.10f ms" % t_random_sample_perm_30)


if __name__ == "__main__":
    test_sample_correctness()
    test_sample_speed()

import torch
import actnn
from actnn.utils import uniform_sample, random_sample_perm
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
    test_implementation(uniform_sample)
    test_implementation(random_sample_perm)


def test_sample_speed():
    print("======== Test Sample Speed ==========")
    data = torch.randn((8000, 512, 768)).cuda()

    def test_implementation(sample_method):
        sample_cnt = 100
        torch.cuda.synchronize()
        start = time.time()
        ret = sample_method(data, sample_cnt)
        torch.cuda.synchronize()
        end = time.time()
        print(ret[0])
        return end - start
    t_uniform_sample = test_implementation(uniform_sample)
    t_random_sample_perm = test_implementation(random_sample_perm)
    print("Uniform sample speed %.10f ms" % t_uniform_sample)
    print("Random sample speed %.10f ms" % t_random_sample_perm)


if __name__ == "__main__":
    test_sample_correctness()
    test_sample_speed()

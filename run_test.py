from multiprocessing import Pool, Queue
import time
import os

q = Queue()

def f(epoch):
    worker_id = q.get()
    os.system('CUDA_VISIBLE_DEVICES={device} ./test_cifar 1 preact_resnet164 {epoch} "-c quantize --bbits 6" 2>&1 | tee {epoch}.log'
              .format(device=worker_id, epoch=epoch))
    q.put(worker_id)

if __name__ == '__main__':
    q.put(4)
    # q.put(5)
    # q.put(6)
    # q.put(7)
    pool = Pool(processes=1)              # start 4 worker processes

    # print "[0, 1, 4,..., 81]"
    pool.map(f, range(1, 201))

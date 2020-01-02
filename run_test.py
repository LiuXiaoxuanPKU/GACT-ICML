from multiprocessing import Pool, Queue
import time
import os

q = Queue()

def f(epoch):
    worker_id, port = q.get()
    os.system('CUDA_VISIBLE_DEVICES={device} ./test_cifar 2 preact_resnet56 {epoch} "-c quantize --bbits 6 --qa=False --persample=True --qw=False" {port} 2>&1 | tee {epoch}.log'
              .format(device=worker_id, epoch=epoch, port=port))
    q.put([worker_id, port])

if __name__ == '__main__':
    q.put(["0,1", 29501])
    q.put(["2,3", 29502])
    #q.put(["4-5", 29503])
    #q.put(["6-7", 29504])
    pool = Pool(processes=2)              # start 4 worker processes

    # print "[0, 1, 4,..., 81]"
    pool.map(f, range(1, 201, 4))


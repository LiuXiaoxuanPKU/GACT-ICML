from multiprocessing import Pool, Queue
import os

host_queue = Queue()
for i in range(0, 8):
    host_queue.put(i)


configs = []
for method in ['naive', 'pg', 'ps', 'pl']:
    if method == 'naive' or method == 'pg':
        bits = [2, 3, 4]
    else:
        bits = [1.5, 1.75, 2, 2.5, 3, 4]

    for bit in bits:
        tid = '{}_{}'.format(method, bit)
        param = '-c quantize --ca=True --cabits={} --calg={}'.format(bit, method)
        configs.append((tid, param))

tasks = []
for seed in range(1):
    for tid, params in configs:
        tasks.append((tid, seed, params))


def launch(task):
    tid, seed, params = task
    host_id = host_queue.get()
    tid = '{}_seed{}'.format(tid, seed)

    work_dir = 'results/{}'.format(tid)
    os.system('mkdir {}'.format(work_dir))
    # cmd = 'CUDA_VISIBLE_DEVICES={hid} python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    # --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 \
    # --gather-checkpoints --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 1 --seed {seed} \
    # {params} ~/data/cifar100 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir,
    #                                                 params=params, tid=tid, seed=seed)
    # cmd = 'CUDA_VISIBLE_DEVICES={hid} python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    # --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 \
    # --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 1 --seed {seed} \
    # --evaluate --training-only --resume results/cifar100/checkpoint-9.pth.tar --resume2 results/cifar100/checkpoint-10.pth.tar \
    # {params} ~/data/cifar100 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir,
    #                                                 params=params, tid=tid, seed=seed)

    cmd = 'CUDA_VISIBLE_DEVICES={hid} ./dist-test 1 0 127.0.0.1 1 resnet50 "{params}" {work_dir} /home/LargeData/Large/ImageNet/ | tee {tid}.log '\
               .format(hid=host_id, work_dir=tid, params=params, tid=tid)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(launch, tasks)

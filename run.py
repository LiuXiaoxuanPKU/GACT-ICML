from multiprocessing import Pool, Queue
import os

prefix='results/20200325_9a367/'

host_queue = Queue()
for i in range(0, 4):
    host_queue.put(i)


tasks = [
    # ('32', '-c quantize --qa=False --qg=False --qw=False'),
    # ('a8w8', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=8'),
    # ('a7w7', '-c quantize --qa=True --qw=True --qg=False --abits=7 --wbits=7'),
    # ('a6w6', '-c quantize --qa=True --qw=True --qg=False --abits=6 --wbits=6'),
    # ('a5w5', '-c quantize --qa=True --qw=True --qg=False --abits=5 --wbits=5'),
    # ('a4w4', '-c quantize --qa=True --qw=True --qg=False --abits=4 --wbits=4'),
    # ('a7w8', '-c quantize --qa=True --qw=True --qg=False --abits=7 --wbits=8'),
    # ('a6w8', '-c quantize --qa=True --qw=True --qg=False --abits=6 --wbits=8'),
    # ('a5w8', '-c quantize --qa=True --qw=True --qg=False --abits=5 --wbits=8'),
    # ('a4w8', '-c quantize --qa=True --qw=True --qg=False --abits=4 --wbits=8'),
    # ('a8w7', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=7'),
    # ('a8w6', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=6'),
    # ('a8w5', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=5'),
    # ('a8w4', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=4'),
    # ('a6w6g4_h', '-c quantize --qa=True --qw=True --qg=True --abits=6 --wbits=6 --bbits=4 --persample=True --hadamard=True'),
    # ('a5w5g4_h', '-c quantize --qa=True --qw=True --qg=True --abits=5 --wbits=5 --bbits=4 --persample=True --hadamard=True'),
    # ('a4w4g4_h', '-c quantize --qa=True --qw=True --qg=True --abits=4 --wbits=4 --bbits=4 --persample=True --hadamard=True'),
    # ('a8w4g4_h', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=4 --persample=True --hadamard=True'),
    # ('a6w6g7', '-c quantize --qa=True --qw=True --qg=True --abits=6 --wbits=6 --bbits=7'),
    # ('a5w5g7', '-c quantize --qa=True --qw=True --qg=True --abits=5 --wbits=5 --bbits=7'),
    # ('a4w4g7', '-c quantize --qa=True --qw=True --qg=True --abits=4 --wbits=4 --bbits=7'),
    # ('a8w4g7', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=7'),
    # ('a6w6g6', '-c quantize --qa=True --qw=True --qg=True --abits=6 --wbits=6 --bbits=6'),
    # ('a5w5g6', '-c quantize --qa=True --qw=True --qg=True --abits=5 --wbits=5 --bbits=6'),
    # ('a4w4g6', '-c quantize --qa=True --qw=True --qg=True --abits=4 --wbits=4 --bbits=6'),
    # ('a8w4g6', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=6'),
    # ('a6w6g5', '-c quantize --qa=True --qw=True --qg=True --abits=6 --wbits=6 --bbits=5'),
    # ('a5w5g5', '-c quantize --qa=True --qw=True --qg=True --abits=5 --wbits=5 --bbits=5'),
    # ('a4w4g5', '-c quantize --qa=True --qw=True --qg=True --abits=4 --wbits=4 --bbits=5'),
    # ('a8w4g5', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=5'),
    ('a8w4g8', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=8 --epochs 200 --lr 0.1'),
    ('a8w4g9', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=9 --epochs 200 --lr 0.1'),
    ('a8w4g7_4x', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=7 --epochs 800 --lr 0.025'),
    ('a8w4g6_4x', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=6 --epochs 800 --lr 0.025'),
    ('a8w4g5_4x', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=5 --epochs 800 --lr 0.025'),
    ('a8w4g4_h_4x', '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits=4 --persample=True --hadamard=True --epochs 800 --lr 0.025'),
]


def launch(task):
    tid, params = task
    host_id = host_queue.get()

    work_dir = prefix + '/' + tid
    cmd = 'mkdir -p {work_dir} && CUDA_VISIBLE_DEVICES={hid} python ./multiproc.py --master_port {hport} \
    --nproc_per_node 1 ./main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    --batch-size 128 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4  \
    {params} ~/data/cifar10'.format(hid=host_id, work_dir=work_dir, hport=29500+host_id, params=params)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(4) as p:
        p.map(launch, tasks)
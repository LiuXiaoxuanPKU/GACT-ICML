from multiprocessing import Pool, Queue
import os

prefix='results/20200325_9a367/'

host_queue = Queue()
for i in range(0, 4):
    host_queue.put(i)


tasks = [
    ('32', '-c quantize --qa=False --qg=False --qw=False'),
    ('a8w8', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=8'),
    ('a7w7', '-c quantize --qa=True --qw=True --qg=False --abits=7 --wbits=7'),
    ('a6w6', '-c quantize --qa=True --qw=True --qg=False --abits=6 --wbits=6'),
    ('a5w5', '-c quantize --qa=True --qw=True --qg=False --abits=5 --wbits=5'),
    ('a4w4', '-c quantize --qa=True --qw=True --qg=False --abits=4 --wbits=4'),
    ('a7w8', '-c quantize --qa=True --qw=True --qg=False --abits=7 --wbits=8'),
    ('a6w8', '-c quantize --qa=True --qw=True --qg=False --abits=6 --wbits=8'),
    ('a5w8', '-c quantize --qa=True --qw=True --qg=False --abits=5 --wbits=8'),
    ('a4w8', '-c quantize --qa=True --qw=True --qg=False --abits=4 --wbits=8'),
    ('a8w7', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=7'),
    ('a8w6', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=6'),
    ('a8w5', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=5'),
    ('a8w4', '-c quantize --qa=True --qw=True --qg=False --abits=8 --wbits=4'),
]


def launch(task):
    tid, params = task
    host_id = host_queue.get()

    work_dir = prefix + '/' + tid
    cmd = 'mkdir -p {work_dir} && CUDA_VISIBLE_DEVICES={hid} python ./multiproc.py --master_port {hport} \
    --nproc_per_node 1 ./main.py --dataset cifar10 --arch preact_resnet56 --gather-checkpoints --workspace {work_dir} \
    --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs 200 \
    {params} ~/data/cifar10'.format(hid=host_id, work_dir=work_dir, hport=29500+host_id, params=params)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(4) as p:
        p.map(launch, tasks)
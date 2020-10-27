from multiprocessing import Pool, Queue
import os

host_queue = Queue()
for i in range(0, 8):
    host_queue.put(i)


tasks = [
    ('exact', '-c fanin'),
    ('4bit', '-c quantize --ca=True --cabits=4 --ibits=4 --persample=False --perlayer=False'),
    ('2bit', '-c quantize --ca=True --cabits=2 --ibits=2 --persample=False --perlayer=False'),
    ('2bit_psr', '-c quantize --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False'),
    ('2bit_ps', '-c quantize --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=False'),
    ('2bit_pl', '-c quantize --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=True'),
    ('2bit_psr_nobn', '-c qlinear --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False'),
    ('2bit_ps_nobn', '-c qlinear --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=False'),
    ('2bit_pl_nobn', '-c qlinear --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=True'),
]

def launch(task):
    tid, params = task
    host_id = host_queue.get()

    work_dir = 'results/{}'.format(tid)
    cmd = 'CUDA_VISIBLE_DEVICES={hid} python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 \
    --gather-checkpoints --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 1 \
    {params} ~/data/cifar10 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir, hport=29500+host_id,
                                                    params=params, tid=tid, epoch=epoch)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(launch, tasks)

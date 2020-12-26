from multiprocessing import Pool, Queue
import os

host_queue = Queue()
for i in range(0, 8):
    host_queue.put(i)


configs = [
    # ('exact', '-c fanin'),
    # ('4bit', '-c quantize --ca=True --cabits=4 --ibits=4 --persample=False --perlayer=False'),
    # ('2bit', '-c quantize --ca=True --cabits=2 --ibits=2 --persample=False --perlayer=False'),
    # # ('2bit_psr', '-c quantize --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False'),
    # ('2bit_ps', '-c quantize --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=False'),
    # ('2bit_pl', '-c quantize --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=True'),
    # ('2bit_psr_nobn', '-c qlinear --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False'),
    # ('2bit_ps_nobn', '-c qlinear --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=False'),
    # ('2bit_pl_nobn', '-c qlinear --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=True'),
    # ('exact_qat', '-c qlinear --qat=8 --ca=True --cabits=8 --ibits=8 --persample=False --perlayer=False'),
    # ('4bit_nobn_qat', '-c qlinear --ca=True --cabits=4 --ibits=4 --persample=False --perlayer=False --qat=8'),
    # ('2bit_nobn_qat', '-c qlinear --ca=True --cabits=2 --ibits=2 --persample=False --perlayer=False --qat=8'),
    # ('2bit_psr_nobn_qat', '-c qlinear --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False --qat=8'),
    ('exact', '-c fanin'),
    ('4bit', '-c quantize --ca=True --cabits=4 --ibits=4 --pergroup=False --perlayer=False'),
    ('2bit', '-c quantize --ca=True --cabits=2 --ibits=2 --pergroup=False --perlayer=False'),
    ('2bit_pg', '-c quantize --ca=True --cabits=2 --ibits=2 --pergroup=True --perlayer=False'),
    ('2bit_pg_ps', '-c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False'),
    ('2bit_pg_pl', '-c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True'),
    ('2bit_pg_ps_nog', '-c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=False --usegradient=False'),
    ('2bit_pg_pl_nog', '-c quantize --ca=True --cabits=2 --ibits=8 --pergroup=True --perlayer=True --usegradient=False'),
]
tasks = []
for tid, params in configs:
    for seed in range(1):
        tasks.append((tid, seed, params))


def launch(task):
    tid, seed, params = task
    host_id = host_queue.get()
    tid = '{}_seed{}'.format(tid, seed)

    work_dir = 'results/{}'.format(tid)
    os.system('mkdir {}'.format(work_dir))
    cmd = 'CUDA_VISIBLE_DEVICES={hid} python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 \
    --gather-checkpoints --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 1 --seed {seed} \
    {params} ~/data/cifar10 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir,
                                                    params=params, tid=tid, seed=seed)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(launch, tasks)

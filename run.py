from multiprocessing import Pool, Queue
import os

prefix='results/20200417_073e2/'

host_queue = Queue()
for i in range(0, 4):
    host_queue.put(['g24', i])
for i in range(0, 8):
    if i != 1 and i != 6:
        host_queue.put(['g32', i])

tasks = []
for seed in range(5):
    tasks.append(('32_seed{}'.format(seed), '-c quantize --qa=False --qg=False --qw=False --seed {}'.format(seed)))
    for bbits in range(4, 9):
        tasks.append(('a8w4g{}_seed{}'.format(bbits, seed), '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits={} --seed {}'.format(bbits, seed)))
        tasks.append(('a8w4g{}_p_seed{}'.format(bbits, seed),
                      '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits={} --persample=True --seed {}'.format(bbits, seed)))
        tasks.append(('a8w4g{}_h_seed{}'.format(bbits, seed),
                      '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=4 --bbits={} --persample=True --hadamard=True --seed {}'.format(
                          bbits, seed)))

def launch(task):
    tid, params = task
    host_name, host_id = host_queue.get()

    work_dir = prefix + '/' + tid
    cmd = 'conda activate nvidia && cd ~/work/RN50v1.5 && mkdir -p {work_dir} && CUDA_VISIBLE_DEVICES={hid} python ./multiproc.py --master_port {hport} \
    --nproc_per_node 1 ./main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} --epochs 200 --lr 0.1 \
    --batch-size 128 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4  \
    {params} ~/data/cifar10'.format(hid=host_id, work_dir=work_dir, hport=29500+host_id, params=params)
    print(cmd)
    os.system("ssh {} << 'ENDSSH'\nsource ~/.zshrc\n{}\nENDSSH".format(host_name, cmd))

    host_queue.put([host_name, host_id])


if __name__ == '__main__':
    with Pool(10) as p:
        p.map(launch, tasks)
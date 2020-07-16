from multiprocessing import Pool, Queue
import os

prefix='/data/chenjianfei/results/20200527/'

host_queue = Queue()
for i in range(0, 8):
    host_queue.put(['Gserver6', i])
for i in range(0, 8):
    host_queue.put(['Gserver3', i])

tasks = []
seed = 0
tasks.append(('a32w32g32_seed{}'.format(seed), '--seed {}'.format(seed)))
tasks.append(('a8w8g32_seed{}'.format(seed), '-c quantize --qa=True --qw=True --qg=False --seed {}'.format(seed)))
for bbits in range(4, 9):
    tasks.append(('a8w8g{}_seed{}'.format(bbits, seed), '-c quantize --qa=True --qw=True --qg=True --bbits={} --seed {}'.format(bbits, seed)))
    tasks.append(('a8w8g{}_p_seed{}'.format(bbits, seed),
                  '-c quantize --qa=True --qw=True --qg=True --bbits={} --persample=True --seed {}'.format(bbits, seed)))
    tasks.append(('a8w8g{}_h_seed{}'.format(bbits, seed),
                  '-c quantize --qa=True --qw=True --qg=True --bbits={} --persample=True --hadamard=True --seed {}'.format(
                      bbits, seed)))

def launch(task):
    tid, params = task
    host_name, host_id = host_queue.get()

    work_dir = prefix + '/' + tid
    cmd = 'conda deactivate && conda activate nvidia && cd ~/RN50v1.5 && mkdir -p {work_dir} && CUDA_VISIBLE_DEVICES={hid} \
    python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} --epochs 200 --lr 0.1 \
    --batch-size 128 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4  \
    {params} ~/data/cifar10'.format(hid=host_id, work_dir=work_dir, params=params)
    print(cmd)
    os.system("ssh {} << 'ENDSSH'\nsource ~/.zshrc\n{}\nENDSSH".format(host_name, cmd))

    host_queue.put([host_name, host_id])


if __name__ == '__main__':
    with Pool(16) as p:
        p.map(launch, tasks)

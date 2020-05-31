from multiprocessing import Pool, Queue
import os

host_queue = Queue()
for i in range(0, 8):
    host_queue.put(i)


tasks = []

for bbits in range(4, 9):
    for epoch in [100]: #[0, 100, 200]:
        tasks.append(['g{}_e{}'.format(bbits, epoch), epoch,
                      '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=8 --bbits={}'.format(bbits)])
        tasks.append(['g{}_e{}_p'.format(bbits, epoch), epoch,
                      '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=8 --bbits={} --persample=True'.format(bbits)])
        tasks.append(['g{}_e{}_h'.format(bbits, epoch), epoch,
                      '-c quantize --qa=True --qw=True --qg=True --abits=8 --wbits=8 --bbits={} --persample=True --hadamard=True'.format(bbits)])


def launch(task):
    tid, epoch, params = task
    host_id = host_queue.get()

    work_dir = 'results/preact_resnet56'
    cmd = 'CUDA_VISIBLE_DEVICES={hid} python ./multiproc.py --master_port {hport} \
    --nproc_per_node 1 ./main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    --resume {work_dir}/checkpoint-{epoch}.pth.tar --evaluate  \
    --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 0 --weight-decay 1e-4 --epochs {epoch_plus_one} \
    {params} ~/data/cifar10 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir, hport=29500+host_id,
                                                    params=params, epoch_plus_one=epoch+1, epoch=epoch, tid=tid)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(launch, tasks)

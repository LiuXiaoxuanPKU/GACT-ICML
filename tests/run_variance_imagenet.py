from multiprocessing import Pool, Queue
import os

host_queue = Queue()
for i in range(0, 8):
    host_queue.put(i)


configs = [
    ('4bit', '-c quantize --ca=True --cabits=4 --ibits=4 --persample=False --perlayer=False'),
    ('2bit', '-c quantize --ca=True --cabits=2 --ibits=2 --persample=False --perlayer=False'),
    ('2bit_psr', '-c quantize --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False'),
    ('2bit_ps', '-c quantize --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=False'),
    ('2bit_pl', '-c quantize --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=True'),
    ('2bit_psr_nobn', '-c qlinear --ca=True --cabits=2 --ibits=2 --persample=True --perlayer=False'),
    ('2bit_ps_nobn', '-c qlinear --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=False'),
    ('2bit_pl_nobn', '-c qlinear --ca=True --cabits=2 --ibits=8 --persample=True --perlayer=True'),
]
tasks = []
for epoch in [1, 84]:
    for tid, params in configs:
        tasks.append((tid, epoch, params))
print(tasks)


def launch(task):
    tid, epoch, params = task
    host_id = host_queue.get()
    tid = 'e{}_{}'.format(epoch, tid)

    work_dir = 'results/resnet50'
    cmd = 'CUDA_VISIBLE_DEVICES={hid} python ./multiproc.py --master_port {hport} \
    --nproc_per_node 1 ./main.py --arch resnet50 --workspace {work_dir} \
    --resume /data2/jianfei/RN50v1.5/resnet/results/resnet50/checkpoint-{epoch}.pth.tar --evaluate  \
    --epochs 91 --batch-size 50 \
    {params} /home/LargeData/Large/ImageNet | tee {tid}.log'.format(hid=host_id, work_dir=work_dir, hport=29500+host_id,
                                                    params=params, tid=tid, epoch=epoch)
    print(cmd)
    os.system(cmd)

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(launch, tasks)

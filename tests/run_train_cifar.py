from multiprocessing import Pool, Queue
import os
import uuid


host_queue = Queue()
for i in range(0, 8):
    host_queue.put(('g14', i))


configs = []
for stochastic in ['False']:
    for method in ['naive', 'pg', 'ps', 'pl']:
        if method == 'naive' or method == 'pg':
            bits = [2, 3, 4]
        else:
            bits = [1.5, 1.75, 2, 2.5, 3, 4]

        for bit in bits:
            tid = '{}_{}_{}'.format(method, bit, stochastic)
            param = '-c quantize --ca=True --sq={} --cabits={} --calg={}'.format(stochastic, bit, method)
            configs.append((tid, param))

tasks = []
for seed in range(1):
    for tid, params in configs:
        tasks.append((tid, seed, params))


def launch(task):
    tid, seed, params = task
    machine_name, host_id = host_queue.get()
    tid = '{}_seed{}'.format(tid, seed)

    work_dir = 'results/{}'.format(tid)

    script_name = 'tmp/' + str(uuid.uuid4())

    with open(script_name, 'w') as f:
        #f.write('source ~/.bashrc\n')
        f.write("""
# added by Anaconda3 5.3.1 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/data1/jianfei/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/data1/jianfei/anaconda3/etc/profile.d/conda.sh" ]; then
# . "/data1/jianfei/anaconda3/etc/profile.d/conda.sh"  # commented out by conda initialize
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/data1/jianfei/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<\n""")
        f.write('conda activate /data1/jianfei/pytorch-1.6\n')
        f.write('cd /data1/jianfei/RN50v1.5/resnet\n')
        f.write('mkdir {}\n'.format(work_dir))
        f.write('CUDA_VISIBLE_DEVICES={hid} python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    --epochs 200 --num-classes 10 -j 0 --weight-decay 1e-4 \
    --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 1 --seed {seed} \
    {params} ~/data/cifar10 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir,
                                                    params=params, tid=tid, seed=seed))

    cmd = "ssh {} 'bash -s' < {}".format(machine_name, script_name)
    os.system(cmd)

    # cmd = 'CUDA_VISIBLE_DEVICES={hid} python main.py --dataset cifar10 --arch preact_resnet56 --workspace {work_dir} \
    # --epochs 200 --num-classes 100 -j 0 --weight-decay 1e-4 \
    # --batch-size 128 --lr 0.1 --momentum 0.9 --label-smoothing 0  --warmup 1 --seed {seed} \
    # --evaluate --training-only --resume results/cifar100/checkpoint-9.pth.tar --resume2 results/cifar100/checkpoint-10.pth.tar \
    # {params} ~/data/cifar100 | tee {tid}.log'.format(hid=host_id, work_dir=work_dir,
    #                                                 params=params, tid=tid, seed=seed)

    # cmd = 'CUDA_VISIBLE_DEVICES={hid} ./dist-test 1 0 127.0.0.1 1 resnet50 "{params}" {work_dir} /home/LargeData/Large/ImageNet/ | tee {tid}.log '\
    #            .format(hid=host_id, work_dir=tid, params=params, tid=tid)
    # print(cmd)
    #

    host_queue.put(host_id)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(launch, tasks)

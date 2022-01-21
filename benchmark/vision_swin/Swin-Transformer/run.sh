python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
	--cfg configs/swin_small_patch4_window7_224.yaml --data-path ~/imagenet --batch-size 32 --get-mem --level L1

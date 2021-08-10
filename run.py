import os

cmd = f'CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ' \
      f'--master_port=8799 train.py  --num_worker 4  --resolution 512  ' \
      f'/home/anpei/code/softgan_test/dataset/ ' \
      f'--with_tensorboard --name test  --iter 800000 --batch 1 --mixing 0.9 --g_reg_every 10000 ' \
      f'--condition_path /mnt/new_disk/chenap/dataset/segmaps/ '
os.system(cmd)
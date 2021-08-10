
# SofGAN (TOG 2021)
## [Project page](https://apchenstu.github.io/sofgan/) |  [Paper](https://arxiv.org/abs/2007.03780)
This repository contains a pytorch implementation for the paper: [SofGAN: A Portrait Image Generator with Dynamic Styling](https://arxiv.org/abs/2007.03780).
We propose a SofGAN image generator to decouple the latent space of portraits into two subspaces: a geometry space and a texture space.
Experiments on SofGAN show that our system can generate high quality portrait images with independently controllable geometry and texture attributes.<br><br>

![Teaser](https://github.com/apchenstu/apchenstu.github.io/blob/master/sofgan/img/semantic_level.png)

## Installation

#### Require Ubuntu 16.04 + Pytorch>=1.2.0 and torchvision>=0.4.0

Install environment:
```
git clone https://github.com/apchenstu/sofgan.git --recursive
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
pip install tqdm argparse scikit-image lmdb config-argparse dlib
```

## Training
Please see each subsection for training on different datasets. Available training datasets:

* [FFHQ](https://github.com/NVlabs/stylegan)
* [CelebA](https://github.com/switchablenorms/CelebAMask-HQ)
* [Your own data](#your-own-data) (portrait images or segmaps)

We also provide our pre-process [ffhq and celeba segmaps](https://drive.google.com/file/d/1_gSENMI5hYj-JTjqtn14PkoLLnEp94oY/view?usp=sharing) (in our classes labels). You may also want to
   retrain the [SOF model](https://github.com/walnut-REE/sof/) base on your own multi-view segmaps.

Run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=9999 train.py \
    --num_worker 4  --resolution 1024
   --name $exp_name
   --iter 10000000
   --batch 1 --mixing 0.9 \
   path/to/your/image/folders \
   --condition_path path/to/your/segmap/folders
```

We strongly suggest you to use at least 4 GPUs, 4GPUs would take  ~20 days for 10000k iterations. You can also reduce the resolution or iterations on your own dataset,
you should have pleasure results after 1000k iterations.

*Important*: training on none pair-wise data (image/segmap) is encouraged, this is one of the key features of our sofgan.

## Rendering
We provide a rendering script in `renderer.ipynb`, you can restyle your own photos, videos, generating free-viewpoing portrait images.
Just to download our  [checkpoints](https://drive.google.com/file/d/1LPKU3AJVlhnyXBGzLS0UrOEhIT1gcFpD/view?usp=sharing) and unzip to the root folder.

## UI Illustration
   A Painter is in included in `Painter`, you can pull down and drawing on-the-fly.
   Before that, you need to install the enviroment with ```pip install -r ./Painter/requirements.txt```

![UI](https://github.com/apchenstu/GIFs/blob/main/sofgan.gif)

## IOS App
You could download and try the [Wand](https://apps.apple.com/cn/app/wand/id1574341319), an ipad app developed by [Deemos](https://www.deemos.com/).

![two-dimensions](https://github.com/apchenstu/GIFs/blob/main/two-dimensions.gif)

## Online Demo
On the way ...

## Relevant Works
[**StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows (TOG 2021)**](https://arxiv.org/abs/2008.02401)<br>
Rameen Abdal, Peihao Zhu, Niloy Mitra, Peter Wonka

[**SEAN: Image Synthesis With Semantic Region-Adaptive Normalization (CVPR 2020)**](https://arxiv.org/abs/1911.12861)<br>
Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka

[**StyleRig: Rigging StyleGAN for 3D Control over Portrait Images (CVPR 2020)**](https://gvv.mpi-inf.mpg.de/projects/StyleRig/)<br>
A. Tewari, M. Elgharib, G. Bharaj, F. Bernard, H.P. Seidel, P. Pérez, M. Zollhöfer, Ch. Theobalt

[**StyleGAN2: Analyzing and Improving the Image Quality of {StyleGAN} (CVPR 2020)**](https://arxiv.org/abs/1912.04958)<br>
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila

[**SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization (CVPR 2019)**](https://arxiv.org/abs/1903.07291)<br>
Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu

## Citation
If you find our code or paper helps, please consider citing:
```
@article{sofgan,
author = {Chen, Anpei and Liu, Ruiyang and Xie, Ling and Chen, Zhang and Su, Hao and Yu Jingyi},
title = {SofGAN: A Portrait Image Generator with Dynamic Styling},
year = {2021},
issue_date = {Jul 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {1},
url = {https://doi.org/10.1145/3470848},
doi = {10.1145/3470848},
journal = {ACM Trans. Graph.},
month = July,
articleno = {1},
numpages = {26},
keywords = {image editing, Generative adversarial networks}
}


'''
# scipy example
#  python test_our.py -i   /root/anpei/code/styleGAN3/dataset/segs -o /root/anpei/code/styleGAN3/result/randomStyle --ckpt /root/anpei/code/styleGAN3/checkpoint/tf-1024-conv-seg-less5-7/279999.pt --MODE 0
#  python test_our.py -i   /root/anpei/code/styleGAN3/dataset/segs -o /root/anpei/code/styleGAN3/result/randomSeg --ckpt /root/anpei/code/styleGAN3/checkpoint/tf-1024-conv-seg-less5-7/279999.pt --MODE 1
#  python test_our.py -i   ./dataset/CCTV-crop.avi -o ./result/video/res.avi --ckpt ./checkpoint/tf-1024-conv-seg-less5-7/279999.pt --MODE 5
#
'''

import cv2
import torch, os, argparse
import numpy as np
from PIL import Image
from scipy.stats import norm
from torch.nn import functional as F
from torchvision import transforms, utils


from modules.model_seg_input import Generator
from modules.BiSeNet import BiSeNet

def vis_condition_img(img):
    part_colors = torch.tensor([[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],#'skin',1 'eye_brow'2,  'eye'3
                    [240, 157, 240], [255, 212, 255], #'r_nose'4, 'l_nose'5
                    [31, 162, 230], [127, 255, 255], [127, 255, 255],#'mouth'6, 'u_lip'7,'l_lip'8
                    [0, 255, 85], [0, 255, 170], #'ear'9 'ear_r'10
                    [255, 255, 170],
                    [127, 170, 255], [85, 0, 255], [255, 170, 127], #'neck'11, 'neck_l'12, 'cloth'13
                    [212, 127, 255], [0, 170, 255],#, 'hair'14, 'hat'15
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]).float()

    N,C,H,W = img.size()
    condition_img_color = torch.zeros((N,3,H,W))
    num_of_class = int(torch.max(img))
    for pi in range(1, num_of_class + 1):
        index = torch.nonzero(img == pi)
        condition_img_color[index[:,0],:,index[:,2], index[:,3]] = part_colors[pi]
    condition_img_color = condition_img_color/255*2.0-1.0
    return condition_img_color

def random_crop(tensor, H=256,W=256, Scale=0.5):

    sample_condition_img = transforms.functional.resize(tensor, (int(H * (1.0+Scale)), int(W * (1.0+Scale))),interpolation=Image.NEAREST)
    sample_condition_img = transforms.ToTensor()(sample_condition_img)
    top_left = [int(np.random.rand() * 0.5 * H), int(np.random.rand() * 0.5 * W)]
    sample_condition_img = sample_condition_img[:,top_left[0]:top_left[0]+H,top_left[1]:W+top_left[1]]

    return sample_condition_img

remap_raw_to_new_list = torch.tensor([0,1,2,3,4,5,14,11,12,13,6,7,8,9,10,15,16,17,18,19]).float()
def id_raw_to_new(seg):
    return remap_raw_to_new_list[seg.long()]


def id_remap(seg):
    remap_list = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16],device=seg.device).float()
    return remap_list[seg.long()]

# remap_list_20_to_17 = torch.tensor([0,1,2,2,3,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16]).float()
# def id_remap(seg):
#     return remap_list[seg.long()]

def get_2d_nose(style_img_size,batch=1,c=1):
    return torch.randn((batch, c, style_img_size, style_img_size)).cuda()

def gen_style_noise(label_img, std, bias, style_img_size=64):
    batch, c, height, width = label_img.size()
    if height != style_img_size or width != style_img_size:
        label_img = F.interpolate(label_img, size=(style_img_size, style_img_size), mode='bilinear')

    noise = get_2d_nose(style_img_size, batch, c)
    label_img = label_img.view((batch, c, style_img_size, style_img_size)).long()
    noise = noise * std[label_img] + bias[label_img]
    return noise

def divide_chunks(l, n):
    return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]

def gaussion(x,sigma=1,mu=0):
    return 1.0/(sigma*np.sqrt(2*np.pi)) * np.exp(- (x-mu)**2/(2*sigma**2))


from modules.model_seg_input import scatter as scatter_model

# ['background'0,'skin'1, 'eye_brow'2, 'eye'3,'r_nose'4, 'l_nose'5, 'mouth'6, 'u_lip'7,
# 'l_lip'8, 'ear'9, 'ear_r'10, 'eye_g'11, 'neck'12, 'neck_l'13, 'cloth'14, 'hair'15, 'hat'16]
# IDList = [[1,4,5,9,12],[6,7,8,3],[11],[16,10], \
#           np.arange(17).tolist(),[15],[14],[0]]
IDList = [np.arange(17).tolist(),[0],[1,4,5,9,12],[15],[6,7,8,3],[11,13,14,16,10]]
# IDList = [[0],[1,4,5,9,12],[15],[2,3,6,7,8,10,11,13,14,16]]
groupName = ['Global','Background','Complexion','Hair','Eyes & Mouth','Wearings']
def scatter_to_mask(segementation, out_num=1,add_whole=True,add_flip=False,region=None):
    segementation = scatter_model(segementation)
    masks = []

    if None == region:
        if add_whole:
            mask = torch.sum(segementation, dim=1, keepdim=True).clamp(0.0, 1.0)
            masks.append(torch.cat((mask, 1.0 - mask), dim=1))
        if add_flip:
            masks.append(torch.cat((1.0 - mask, mask), dim=1))


        for i in range(out_num - add_whole - add_flip):
            idList = IDList[i]
            mask = torch.sum(segementation[:, idList], dim=1, keepdim=True).clamp(0.0, 1.0)
            masks.append(torch.cat((1.0 - mask, mask), dim=1))
    else:
        for item in region:
            idList = IDList[item]
            mask = torch.sum(segementation[:, idList], dim=1, keepdim=True).clamp(0.0, 1.0)
            masks.append(torch.cat((1.0 - mask, mask), dim=1))
    masks = torch.cat(masks, dim=0)
    return masks

def scatter_to_mask_random(segementation, out_num=1,add_whole=True,add_flip=False,idx=None):
    segementation = scatter_model(segementation)
    masks = []

    if add_whole:
        mask = torch.sum(segementation, dim=1, keepdim=True).clamp(0.0, 1.0)
        masks.append(torch.cat((mask, 1.0 - mask), dim=1))
    if add_flip:
        masks.append(torch.cat((1.0 - mask, mask), dim=1))


    for i in range(out_num - add_whole - add_flip):
        if idx is None:
            idList = IDList[np.random.randint(0,len(IDList))]
        else:
            idList = IDList[idx[i]]
        mask = torch.sum(segementation[:, idList], dim=1, keepdim=True).clamp(0.0, 1.0)
        masks.append(torch.cat((1.0 - mask, mask), dim=1))

    masks = torch.cat(masks, dim=0)
    return masks


import random
def make_noise(batch, styles_dim, style_repeat, latent_dim, n_noise, device):
    noises = torch.randn(n_noise, batch, styles_dim, latent_dim, device=device).repeat(1,1,style_repeat,1)
    return noises

def mixing_noise(batch, latent_dim, prob, device, unbine=True):
    n_noise = 1
    style_dim = 2 if random.random() < prob else 1
    style_repeat = 2//style_dim #if prob>0 else 1
    styles = make_noise(batch, style_dim, style_repeat, latent_dim, n_noise, device)
    return styles.unbind(0)if unbine else styles

def gen_noise(shape):
    return torch.randn(shape).cuda()


@torch.no_grad()
def to_w_style(S, style, av, trunc_psi=0.6, segmap=None):
    w_space = []

    for tensor in style:
        tmp = S(tensor)
        tmp = trunc_psi * (tmp - av) + av
        w_space.append(tmp)

    return w_space

@torch.no_grad()
def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model.style_map_norepeat(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

@torch.no_grad()
def cal_av(generator, batch_size, latent_dim):
    z = gen_noise((4000, latent_dim))
    samples = evaluate_in_chunks(batch_size, generator, z)
    av = torch.mean(samples, dim=0)[None]
    return av

def initFaceParsing(n_classes=20):
    net = BiSeNet(n_classes)
    net.cuda()
    net.load_state_dict(torch.load('modules/segNet-20Class.pth'))
    net.eval()
    return net

def scatter(codition_img,source=None, classSeg=20):
    batch, c, height, width = codition_img.size()
    input_label = torch.cuda.BoolTensor(batch, classSeg, height,width).zero_()

    if source is None:
        source = 1
    return input_label.scatter_(1, codition_img.long(),source)

def mIOU(source,target):
    source, target = scatter(source), scatter(target)
    mIOU = torch.mean(torch.div(
        torch.sum(source & target, dim=[2,3]).float(),
        torch.sum(source | target, dim=[2,3]).float()+1e-6),dim=1)
    return mIOU

def sample_segmap_from_list(folder_img, img_list, samples=4):
    segmaps = []
    idx = np.random.randint(0,len(img_list),samples)
    for item in idx:
        img_path = os.path.join(folder_img, img_list[item])
        seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()
        seg_label = id_remap(F.interpolate(seg_label, size=(128, 128), mode='nearest') * 255)
        segmaps.append(seg_label)
    segmaps = torch.cat(segmaps).cuda()
    return segmaps

def sample_styles_with_miou(seg_label,num_style,mixstyle=0, truncation=0.9, batch_size=4, descending=False):
    times = 0
    in_batch = seg_label.shape[0]
    if in_batch == 1:
        batch = batch_size
        seg_label = seg_label.repeat(batch,1,1,1)
    else:
        batch = in_batch

    with torch.no_grad():
        styles_miou,count,mious = [],0,[]
        while count < num_style:
            styles = mixing_noise(batch//in_batch, args.latent, mixstyle, device, unbine=False)
            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=truncation)
            styles = torch.cat(styles, dim=0)
            w_latent = generator.style_map([styles], to_w_space=False)

            if in_batch > 1:
                w_latent = w_latent.repeat(batch,1,1)

            img, _, _, _ = generator(w_latent, return_latents=False, condition_img=seg_label, input_is_latent=True, noise=noise)
            img = img.clamp(-1.0, 1.0)
            img = F.interpolate(img, size=(512, 512), mode='bilinear')

            segmap = bisNet(img)[0]
            segmap = F.interpolate(segmap, size=seg_label.shape[2:], mode='bilinear')
            segmap = id_remap(torch.argmax(segmap, dim=1, keepdim=True))

            thread = 0.46
            if times>15:
                thread = 0.42
            if times>20:
                thread = 0.35
            if times>30:
                thread = 0.

            miou = mIOU(segmap, seg_label)
            miou = miou.min() if in_batch>1 else miou
            mask = (miou > thread).tolist()

            times += 1
            if np.sum(mask) == 0:
                continue

            if in_batch > 1 and mask:
                mious.append(miou.view(-1,1))
                styles_miou.append(w_latent[[0]])
                count += 1
            else:
                mious.append(miou[mask])
                if len(mask) == w_latent.shape[0]:
                    styles_miou.append(w_latent[mask])
                else:
                    styles_miou.append(w_latent.view(-1,2,w_latent.shape[-2],w_latent.shape[-1])[mask])  # old need this
                count += np.sum(mask)

    mious = torch.cat(mious, dim=0).view(-1)
    mious, indices = torch.sort(mious, descending=descending)
    styles_miou = torch.cat(styles_miou,dim=0)[indices]
    return styles_miou[:num_style]



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-batch_size', type=int,default=4)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--nrows', type=int, default=6)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--with_rgb_input', action='store_true')
    parser.add_argument('--with_local_style', action='store_true')
    parser.add_argument('--condition_dim', type=int, default=0)
    parser.add_argument('--styles_path', type=str, default=None)
    parser.add_argument('--MODE', type=int, default=0)
    parser.add_argument('--miou_filter', action='store_true')
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--with_seg_fc', action='store_true')
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    args.condition_path = args.input
    generator = Generator2(args).to(device)
    generator.eval()


    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt)
        generator.load_state_dict(ckpt['g_ema'])

        del ckpt
        torch.cuda.empty_cache()


    transform = transforms.Compose(
        [
            # transforms.RandomAffine(20,translate=(0.2,0.2)),
            transforms.ToTensor(),
        ]
    )

    if os.path.isdir(args.input):
        List = sorted(os.listdir(args.input))

    if not os.path.exists(args.output):
        os.mkdir(os.path.abspath(args.output))

    batch_size = 4
    latent_av = cal_av(generator, batch_size, args.latent)
    # latent_av = np.load('center_gender.npy', allow_pickle=True)[()]
    # print(np.abs(latent_av['center_man']-latent_av['center_woman']))
    # latent_av = latent_av['center_woman'].reshape((1,512))
    # print(latent_av.shape)

    # seg
    bisNet = initFaceParsing()

    resolution_vis = 512
    # random style
    if 0 == args.MODE:
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            for i, item in enumerate(List[:1000]):
                if (not item.endswith('g')) or os.path.isdir(os.path.join(args.input,item)):
                    continue
                print('Processing image: %s of %d/%d'%(item,i,len(List)))
                img_path = os.path.join(args.input,item)
                save_path = os.path.join(args.output,item)
                seg_label = Image.open(img_path)
                seg_label = transform(seg_label).unsqueeze(0).cuda()
                # seg_label = random_crop(seg_label).unsqueeze(0).cuda()

                seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)
                # seg_label = torch.round(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)

                # result.append(vis_condition_img(seg_label))


                mixstyle = 0.0
                # if i%6==0:

                if not args.miou_filter:
                    styles = mixing_noise(args.nrows, args.latent, mixstyle, device, unbine=False)
                    styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                    styles = torch.cat(styles, dim=0)
                    w_latent = generator.style_map([styles], to_w_space=False)
                    w_latent = w_latent.view(-1,2,w_latent.shape[-2],w_latent.shape[-1])
                else:
                    w_latent = sample_styles_with_miou(seg_label, args.nrows * 2, mixstyle=mixstyle,
                                                   truncation=args.truncation, batch_size=args.batch_size)


                for j in range(args.nrows):
                    fake_img, _, _, _ = generator(w_latent[j], return_latents=False, condition_img=seg_label.cuda(), \
                                                  input_is_latent=True, noise=noise)
                    result.append(fake_img.detach().cpu())


                # if args.nrows-1 == i%args.nrows or i==len(List)-1:
                result = torch.cat(result, dim=0)
                result = F.interpolate(result,(resolution_vis,resolution_vis))
                utils.save_image(
                    result,
                    os.path.join(args.output,'%s'%item),
                    nrow=args.nrows,
                    normalize=True,
                    range=(-1, 1),
                    padding = 0
                )
                count += 1
                result = []

    # local style
    elif 1 == args.MODE:
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            for i, item in enumerate(List):
                if not item.endswith('g'):
                    continue
                print('Processing image: %s of %d/%d'%(item,i,len(List)))
                img_path = os.path.join(args.input,item)
                save_path = os.path.join(args.output,item)

                seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()
                seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)
                # seg_label = F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255

                result.append(vis_condition_img(seg_label))

                seg_label = seg_label.cuda()
                styles = mixing_noise(1, args.latent, 1.0, device, unbine=False)

                styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                styles = torch.cat(styles, dim=0)
                w_latent = generator.style_map([styles], to_w_space=False)
                style_masks = scatter_to_mask(seg_label, args.nrows, add_flip=True)

                for j in range(args.nrows):
                    fake_img, _, _, _ = generator(w_latent, return_latents=False, condition_img=seg_label, \
                                                  input_is_latent=True, noise=noise, style_mask = style_masks[[j]])
                    fake_img = fake_img.clamp(-1.0, 1.0)
                    result.append(fake_img.detach().cpu())

                if args.nrows-1 == i%args.nrows or i==len(List)-1:
                    result = torch.cat(result, dim=0)
                    result = F.interpolate(result,(resolution_vis,resolution_vis))
                    utils.save_image(
                        result,
                        os.path.join(args.output,'%06d.png'%count),
                        nrow=args.nrows + 1,
                        normalize=True,
                        range=(-1, 1),
                    )
                    count += 1
                    result = []

    # multi view animation
    elif 2 == args.MODE:
        resolution_vis = 512
        nrows,ncols = 5,5
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            # visList = os.listdir(os.path.join(args.input,'vis'))
            for i, folder in enumerate(List):

                if folder=='vis'  :#or (folder+'.gif') not in visList
                    continue

                print('Processing folder: %s of %d/%d' % (folder, i, len(List)))
                folder_img = os.path.join(args.input,folder)#,'seg'
                img_list = sorted(os.listdir(folder_img))

                width_pad = 0#2 * (ncols + 1) if ncols > 1 or nrows>1 else 0
                height_pad = 0#2 * (nrows + 1) if nrows > 1 or ncols>1 else 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % folder), fourcc,
                                      20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))


                count = 0
                style_idx = np.random.randint(0,len(IDList),nrows*ncols)
                for k,item in enumerate(img_list):
                    if not item.endswith('g') or os.path.isdir(os.path.join(folder_img,item)): #or (k>=180 and k<290):
                        continue

                    img_path = os.path.join(folder_img, item)
                    save_path = os.path.join(folder_img, item)
                    seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()

                    result = []
                    seg_label = id_remap(F.interpolate(seg_label, size=(resolution_vis, resolution_vis), mode='nearest') * 255)
                    # result.append(vis_condition_img(seg_label))
                    seg_label = seg_label.cuda()

                    if count==0:
                        mixstyle = 1.0
                        if not args.miou_filter:
                            styles = mixing_noise(nrows * ncols, args.latent, mixstyle, device, unbine=False)
                            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                            styles = torch.cat(styles, dim=0)
                            w_latent = generator.style_map([styles], to_w_space=False)
                        else:
                            w_latent = sample_styles_with_miou(seg_label,nrows * ncols*2, mixstyle=mixstyle, truncation=args.truncation, batch_size=args.batch_size)
                        print('==> sample styles done.')

                    n_sample = ncols*nrows
                    batch_sizes = divide_chunks(list(range(n_sample)),args.batch_size)
                    style_masks = scatter_to_mask_random(seg_label, nrows*ncols, add_flip=True, idx=style_idx)

                    for batch in batch_sizes:
                        fake_img, _, _, _ = generator(w_latent[batch], return_latents=False, condition_img=seg_label.repeat(len(batch),1,1,1), \
                                                      input_is_latent=True, noise=noise, style_mask=style_masks[batch])
                        fake_img = fake_img.clamp(-1.0, 1.0)
                        fake_img = F.interpolate(fake_img, size=(resolution_vis, resolution_vis), mode='nearest')
                        result.append(fake_img.detach().cpu())

                    result = torch.cat(result, dim=0)
                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                    result = (utils.make_grid(result, nrow=ncols, padding=0) + 1) / 2 * 255
                    # cv2.imwrite(args.output+'/%s_%02d.png'%(folder,k),(result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0)))
                    out.write((result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0)))
                    count += 1


    # random style animation
    # CUDA_VISIBLE_DEVICES=1 python test_our2.py -i ./dataset/segs/ -o ./result/dynamic_styles --ckpt ./checkpoint/slow-stylegan2-face-tf-1024-spade-64-256-seg-encoder-128-16-random-crop/199999.pt --MODE 3
    elif 3 == args.MODE:
        with torch.no_grad():
            nrows,ncols = 2,3
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

            for i, item in enumerate(List):
                if not item.endswith('g'):
                    continue
                print('Processing image: %s of %d/%d'%(item,i,len(List)))
                img_path = os.path.join(args.input,item)
                save_path = os.path.join(args.output,item)
                seg_label = Image.open(img_path)
                seg_label = transform(seg_label).unsqueeze(0).cuda()
                # seg_label = random_crop(seg_label).unsqueeze(0).cuda()

                seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)
                # seg_label = torch.round(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)

                seg_label_rgb = vis_condition_img(seg_label)
                result.append(seg_label_rgb)

                # w_latent = generator.style_map(styles)
                width_pad = 2 * (ncols + 1) if ncols > 1 else 0
                height_pad = 2 * (nrows + 1) if nrows > 1 else 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % item[:-4]), fourcc,
                                      20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))

                styles_count = nrows*ncols - 1
                styles_last = mixing_noise(styles_count, args.latent, 0.9, device, unbine=False)
                styles_last = to_w_style(generator.style_map_norepeat, styles_last, latent_av, trunc_psi=args.truncation)
                styles_last = torch.cat(styles_last, dim=0)

                for frame in range(20):
                    styles_current = mixing_noise(styles_count, args.latent, 0.8, device, unbine=False)
                    styles_current = to_w_style(generator.style_map_norepeat, styles_current, latent_av, trunc_psi=args.truncation)
                    styles_current = torch.cat(styles_current, dim=0)

                    frame_sub_count = 20
                    for frame_sub in range(frame_sub_count):
                        weight = frame_sub/(frame_sub_count-1)

                        for j in range(styles_count):
                            style = weight * styles_current[2 * j:2 * j + 1] + (1.0 - weight) * styles_last[2 * j:2 * j + 1]
                            w_latent = generator.style_map([style], to_w_space=False)
                            fake_img, _, _, _ = generator(w_latent, return_latents=False, condition_img=seg_label.cuda(), \
                                                          input_is_latent=True, noise=noise)
                            result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))

                        result = torch.cat(result, dim=0)
                        result = F.interpolate(result,(resolution_vis,resolution_vis))
                        result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
                        out.write((result.numpy().astype('uint8')[[2, 1, 0]]).transpose((1, 2, 0)))

                        result = [seg_label_rgb]

                    styles_last = styles_current

    # random style animation multi seg
    elif 4 == args.MODE:
        resolution_vis = 1024
        with torch.no_grad():
            args.nrows = 2
            seg_label_rgb,seg_labels = [],[]

            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            for i, item in enumerate(List):
                if not item.endswith('g'):
                    continue
                # print('Processing image: %s of %d/%d'%(item,i,len(List)))
                img_path = os.path.join(args.input,item)
                save_path = os.path.join(args.output,item)
                seg_label = Image.open(img_path)
                seg_label = transform(seg_label).unsqueeze(0).cuda()
                # seg_label = random_crop(seg_label).unsqueeze(0).cuda()

                seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)
                # seg_label = torch.round(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)

                # seg_label_rgb = vis_condition_img(seg_label)
                # result.append(seg_label_rgb)
                seg_labels.append(seg_label)


                nrows, ncols = 1,1
                styles_count = 11
                total_count = nrows * ncols
                if len(seg_labels) == total_count:

                    print('====> Processing frame %s.' % item)
                    width_pad = 2 * (ncols+1) if ncols>1 else 0
                    height_pad = 2 * (nrows + 1) if nrows > 1 else 0
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4'% item[:-4]), fourcc, 20, \
                                          (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))

                    # out_mask = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % item[:-4]),
                    #                       fourcc, 20, (resolution_vis , resolution_vis))

                    mixstyle = 1.0
                    if not args.miou_filter:
                        styles = mixing_noise(styles_count, args.latent, mixstyle, device, unbine=False)
                        styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                        styles = torch.cat(styles, dim=0)
                        w_latent = generator.style_map([styles], to_w_space=False)
                    else:
                        w_latent = sample_styles_with_miou(seg_label,styles_count*2, mixstyle=0.0, truncation=args.truncation, batch_size=args.batch_size)


                    seg_label = torch.cat(seg_labels, dim=0).cuda()
                    style_masks = scatter_to_mask(seg_label, 1+len(IDList), add_flip=False)  # [total_count:]
                    # goodList = np.array([[1,25,20,41,15,6,24,40,31,48,44]])
                    # w_latent = torch.load('%s.th'%item[:-4])[:100].view((-1,2,14,512))[goodList].view((-1,14,512))
                    # torch.save(w_latent,'%s.th'%item[:-4])
                    for frame in range(styles_count-1):

                        frame_sub_count = 40
                        cdf_scale = 1.0/(1.0-norm.cdf(-frame_sub_count//2,0,6)*2)
                        for frame_sub in range(-frame_sub_count//2,frame_sub_count//2+1):
                            result = []
                            weight = (norm.cdf(frame_sub,0,6)-norm.cdf(-frame_sub_count//2,0,6))*cdf_scale
                            w_latent_current = (1.0 - weight) * w_latent[2*frame:2*frame+2] + weight * w_latent[2*(frame+1):2*(frame+1)+2]

                            n_sample = total_count
                            batch_sizes = divide_chunks(list(range(n_sample)), 1)

                            for batch in batch_sizes:
                                fake_img, _, _, _ = generator(w_latent_current, return_latents=False, condition_img=seg_label[batch], \
                                                              input_is_latent=True, noise=noise, style_mask=style_masks[batch])
                                result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))


                            result = torch.cat(result, dim=0)
                            result = F.interpolate(result,(resolution_vis,resolution_vis))
                            result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
                            result = (result.detach().numpy()[[2, 1, 0]]).transpose((1, 2, 0))
                            img = np.zeros(result.shape)
                            img[:] = result
                            # result = 0.6*cv2.putText(img, 'Global Styles', org=(img.shape[1]//40, img.shape[0]//20),
                            #                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            #                            fontScale=1.2, color=(255, 255, 255), thickness=img.shape[0]//400) + 0.4 * result
                            out.write(result.astype('uint8'))


                    if args.with_local_style:
                        w_latent_current[1] = w_latent_current[0]
                        w_latent_init = w_latent_current[[0]].clone()

                        for k in range(len(IDList)):
                            dynamic_count = 8
                            for frame in range(dynamic_count):
                                if frame==dynamic_count-1:
                                    w_next = w_latent_init
                                else:
                                    w_next = w_latent[[2*frame]]
                                w_latent_last = w_latent_current[[1]]

                                frame_sub_count = 30
                                cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, 6) * 2)
                                for frame_sub in range(-frame_sub_count//2,frame_sub_count//2+1):
                                    result = []
                                    weight = (norm.cdf(frame_sub,0,6)-norm.cdf(-frame_sub_count//2,0,6))*cdf_scale
                                    w_latent_current = weight * w_next + (1.0 - weight) * w_latent_last
                                    w_latent_current = torch.cat((w_latent_init,w_latent_current),dim=0)
                                    for j in range(total_count):
                                        fake_img, _, _, _ = generator(w_latent_current, return_latents=False, style_mask=style_masks[[(k+1)*total_count+j]],
                                                condition_img=seg_label[[j]], input_is_latent=True, noise=noise)
                                        result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))

                                    result = torch.cat(result, dim=0)
                                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                                    result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
                                    result = (result.numpy()[[2, 1, 0]]).transpose((1, 2, 0))
                                    # img = np.zeros(result.shape)
                                    # img[:] = result
                                    # result =  0.6*cv2.putText(img, groupName[k], org=(img.shape[1]//40, img.shape[0]//20),
                                    #                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    #                    fontScale=1.2, color=(255, 255, 255), thickness=img.shape[0]//400) + 0.4 * result
                                    out.write(result.astype('uint8'))
                    seg_labels = []


    # test on animate video
    elif 5 == args.MODE:
        def parsing_img(bisNet, img, to_tensor, argmax=True):
            with torch.no_grad():
                image = Image.fromarray(img)
                image = image.resize((512, 512), Image.BILINEAR)
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                parsing = bisNet(img)[0]
                if argmax:
                    parsing = parsing.argmax(1, keepdim=True).float()
                # parsing = inpaint(filtting(parsing))
                # parsing = id_remap(parsing)
            return parsing

        remap_list = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16]).float()
        def id_remap(seg):
            return remap_list[seg.long()]

        def initFaceParsing():
            net = BiSeNet(n_classes=17)
            net.cuda()
            net.load_state_dict(torch.load('modules/segNet-17Class.pth'))
            # net.load_state_dict(torch.load('E:/face2face2/code/face-parsing/res/cp/79999_iter.pth'))
            net.eval()
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

            ])
            return net, to_tensor

        with torch.no_grad():
            args.nrows = 3

            bisNet, to_tensor = initFaceParsing()
            cap = cv2.VideoCapture(args.input)
            name = os.path.basename(args.input)
            out = cv2.VideoWriter(os.path.join(args.output), cv2.VideoWriter_fourcc(*'XVID'),25, \
                                  (resolution_vis * args.nrows + 2 * (args.nrows + 1),resolution_vis * args.nrows + 2 * (args.nrows + 1)))

            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

            styles_count = args.nrows ** 2 - 1
            styles = mixing_noise(styles_count, args.latent, 0.9, device, unbine=False)
            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
            styles = torch.cat(styles, dim=0)
            success, img = cap.read()
            while success:
                img = cv2.resize(img, (512, 512))
                seg_label = parsing_img(bisNet, img[..., ::-1], to_tensor)
                seg_label_rgb = vis_condition_img(seg_label)
                seg_label_rgb = F.interpolate(seg_label_rgb, (args.resolution, args.resolution), mode='bilinear', align_corners=True)

                result.append(seg_label_rgb)
                for j in range(args.nrows**2-1):

                    style = styles[2*j:2*j+1]
                    w_latent = generator.style_map([style], to_w_space=False)
                    fake_img, _, _, _ = generator(w_latent, return_latents=False,
                                                  condition_img=seg_label.cuda(), input_is_latent=True, noise=noise)
                    result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))

                result = torch.cat(result, dim=0)
                result = F.interpolate(result, (resolution_vis, resolution_vis))
                result = (utils.make_grid(result, nrow=args.nrows) + 1) / 2 * 255
                result = result[[2, 1, 0]].numpy().transpose((1, 2, 0)).astype('uint8')
                out.write(result)
                result = []
                success, img = cap.read()

    # given style
    if 6 == args.MODE:
        ncols,nrows = 2,1
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            for i, folder in enumerate(List):
                print('Processing image: %s of %d/%d' % (folder, i, len(List)))
                style = torch.load(os.path.join(args.styles_path,folder+'.npy'))
                files = sorted(os.listdir(os.path.join(args.input,folder)))

                out = cv2.VideoWriter(os.path.join(args.output, '%s.avi' % folder),
                                      cv2.VideoWriter_fourcc(*'XVID'), 24, \
                                      (resolution_vis * ncols + 2 * (ncols+1), resolution_vis * nrows + 2 * (nrows+1)))

                for item in files:
                    if not item.endswith('g'):
                        continue
                    result = []
                    img_path = os.path.join(args.input, folder, item)
                    seg_label = Image.open(img_path)
                    seg_label = transform(seg_label).unsqueeze(0).cuda()
                    seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255)
                    result.append(vis_condition_img(seg_label))

                    # w_latent,noise, class_probability = styles[j]['latent_n'], styles[j]['noises'], styles[j]['class_probability']#.unsqueeze(0).repeat((2,1,1))
                    # style_mask = class_probability[seg_label.long()]
                    # style_mask = torch.cat((style_mask,1.0-style_mask),1)
                    # fake_img, _, _, _ = generator([w_latent], return_latents=False, condition_img=seg_label.cuda(), \
                    #                               input_is_latent=False, noise=noise, style_mask=style_mask)
                    #

                    w_latent,noise = style['latent_n'].unsqueeze(0).repeat((2,1,1)), style['noises']
                    fake_img, _, _, _ = generator(w_latent, return_latents=False, condition_img=seg_label.cuda(), \
                                                  input_is_latent=True, noise=noise)

                    result.append(fake_img.clamp(-1,1).detach().cpu())
                    result = torch.cat(result, dim=0)
                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                    result = (utils.make_grid(result, nrow=nrows*ncols) + 1) / 2 * 255
                    result = result[[2, 1, 0]].numpy().transpose((1, 2, 0)).astype('uint8')
                    out.write(result)


    # abalation study for mix style training
    elif 7 == args.MODE:
        IDList,resolution_vis = [[1,4,5,9,12]], 1024  # ,[11],[13]
        with torch.no_grad():
            args.nrows = 2
            seg_label_rgb, seg_labels = [], []

            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            for i, item in enumerate(List):
                if not item.endswith('g'):
                    continue
                # print('Processing image: %s of %d/%d'%(item,i,len(List)))
                img_path = os.path.join(args.input, item)
                save_path = os.path.join(args.output, item)
                seg_label = Image.open(img_path)
                seg_label = transform(seg_label).unsqueeze(0).cuda()
                # seg_label = random_crop(seg_label).unsqueeze(0).cuda()

                seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255)
                # seg_label = torch.round(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)

                # seg_label_rgb = vis_condition_img(seg_label)
                # result.append(seg_label_rgb)
                seg_labels.append(seg_label)

                styles_count = 1
                nrows, ncols = 1,3
                total_count = nrows * ncols
                if len(seg_labels) == total_count:

                    print('====> Processing frame %03d.' % (i // total_count))
                    out = cv2.VideoWriter(os.path.join(args.output, '%03d.avi' % (i // total_count)),
                                          cv2.VideoWriter_fourcc(*'XVID'), 10, \
                                          (resolution_vis * ncols + 2 * (ncols + 1),
                                           resolution_vis * nrows + 2 * (nrows + 1)))

                    styles_last = mixing_noise(styles_count, args.latent, 1.0, device, unbine=False)
                    styles_last = to_w_style(generator.style_map_norepeat, styles_last, latent_av, trunc_psi=args.truncation)
                    styles_last = torch.cat(styles_last, dim=0)

                    seg_label = torch.cat(seg_labels, dim=0).cuda()
                    style_masks = scatter_to_mask(seg_label, 1 + len(IDList), add_flip=False)  # [total_count:]

                    styles_last[[1]] = styles_last[[0]]
                    w_latent_init = generator.style_map([styles_last[[0]]], to_w_space=False)
                    style_init = styles_last.clone()

                    for k in range(len(IDList)):
                        dynamic_count = 20
                        for frame in range(dynamic_count):
                            if frame == dynamic_count - 1:
                                styles_current = style_init
                            else:
                                styles_current = mixing_noise(styles_count, args.latent, 1.0, device, unbine=False)
                                styles_current = to_w_style(generator.style_map_norepeat, styles_current,
                                                            latent_av, trunc_psi=args.truncation)
                                styles_current = torch.cat(styles_current, dim=0)

                            frame_sub_count = 20
                            for frame_sub in range(frame_sub_count):
                                result = []
                                weight = frame_sub / (frame_sub_count - 1)
                                style = weight * styles_current[[1]] + (1.0 - weight) * styles_last[[1]]
                                w_latent = generator.style_map([style], to_w_space=False)
                                w_latent = torch.cat((w_latent_init, w_latent), dim=0)
                                for j in range(total_count):
                                    fake_img, _, _, _ = generator(w_latent, return_latents=False,
                                                                  style_mask=style_masks[[(k + 1) * total_count + j]],
                                                                  condition_img=seg_label[[j]],
                                                                  input_is_latent=True, noise=noise)
                                    result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))

                                result = torch.cat(result, dim=0)
                                result = F.interpolate(result, (resolution_vis, resolution_vis))
                                result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
                                result = (result.numpy()[[2, 1, 0]]).transpose((1, 2, 0))
                                img = np.zeros(result.shape)
                                img[:] = result
                                result = 0.6 * cv2.putText(img, groupName[k],
                                                           org=(img.shape[1] // 40, img.shape[0] // 20),
                                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                           fontScale=1.2, color=(255, 255, 255),
                                                           thickness=img.shape[0] // 400) + 0.4 * result
                                out.write(result.astype('uint8'))

                            styles_last = styles_current
                    seg_labels = []

    # local style
    elif 8 == args.MODE:
        resolution_vis = 512
        with torch.no_grad():
            seg_label_rgb = []
            args.nrows = len(IDList)+2
            result, count, subframe = [], 0, 7
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            for i, item in enumerate(List):
                if not item.endswith('g'):
                    continue
                print('Processing image: %s of %d/%d'%(item,i,len(List)))
                img_path = os.path.join(args.input,item)
                save_path = os.path.join(args.output,item)

                seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()
                seg_label = F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255
                # seg_label = F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255

                result.append(vis_condition_img(seg_label))

                seg_label = seg_label.cuda()
                styles = mixing_noise(subframe*2, args.latent, 1.0, device, unbine=False)

                styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                styles = torch.cat(styles, dim=0)
                w_latent = generator.style_map([styles], to_w_space=False)
                style_masks = scatter_to_mask(seg_label, args.nrows, add_flip=True)

                # for g in range(10):
                for j in range(1, args.nrows):
                    subframe_sub = subframe if j > 1 else 1
                    for k in range(subframe_sub):
                        w_latent_sub = w_latent[[1,k+1]].clone()
                        fake_img, _, _, _ = generator(w_latent_sub, return_latents=False, condition_img=seg_label, \
                                                      input_is_latent=True, noise=noise, style_mask = style_masks[[j]])
                        fake_img = F.interpolate(fake_img, (resolution_vis, resolution_vis))
                        result.append(fake_img.detach().cpu())

                    for k,res in enumerate(result):
                        utils.save_image(
                            res,
                            os.path.join(args.output,'%s_%03d_%03d.png'%(item[:-4],j,k)),
                            nrow=1,
                            normalize=True,
                            range=(-1, 1),
                        )
                    result = []

    elif 9 == args.MODE: # projection vis
        with torch.no_grad():
            resolution_vis = 512
            seg_label_rgb = []
            result, count = [], 0
            files = sorted(os.listdir(os.path.join(args.input , List[0])))
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            for i, item in enumerate(files):

                styles = mixing_noise(1, args.latent, 0.9, device, unbine=False)
                styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                styles = torch.cat(styles, dim=0)

                style = styles[[0]]
                w_latent = generator.style_map([style], to_w_space=False)
                result = []
                print('Processing image: %s of %d/%d'%(item,i,len(List)))
                for j, folder in enumerate(List):
                    img_path = os.path.join(args.input, folder, item)
                    seg_label = Image.open(img_path)
                    seg_label = transform(seg_label).unsqueeze(0).cuda()
                    # seg_label = random_crop(seg_label).unsqueeze(0).cuda()

                    seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)
                    # seg_label = torch.round(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest')*255)

                    result.append(vis_condition_img(seg_label))
                    fake_img, _, _, _ = generator(w_latent, return_latents=False, condition_img=seg_label.cuda(), \
                                                  input_is_latent=True, noise=noise)
                    result.append(fake_img.detach().cpu())

                for k, img in enumerate(result):
                    img = F.interpolate(img, (resolution_vis, resolution_vis))
                    save_path = os.path.join(os.path.join(args.output, item[:-4]+'_%02d.png'%k))
                    utils.save_image(
                        img,
                        save_path,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                # result = torch.cat(result, dim=0)
                # save_path = os.path.join(args.output, item)
                # result = F.interpolate(result,(resolution_vis,resolution_vis))
                # utils.save_image(
                #     result,
                #     save_path,
                #     nrow=len(List)*2,
                #     normalize=True,
                #     range=(-1, 1),
                # )
                # count += 1

    # multi view animation two rotate
    elif 10 == args.MODE:
        nrows,ncols = 1,1
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            visList = os.listdir(os.path.join(args.input,'vis'))
            for i, folder in enumerate(List):
                if folder=='vis' or (folder+'.gif') not in visList:
                    continue

                print('Processing folder: %s of %d/%d' % (folder, i, len(List)))
                folder_img = os.path.join(args.input,folder)#,'seg'
                img_list = sorted(os.listdir(folder_img))

                width_pad = 2 * (ncols + 1) if ncols > 1 or nrows>1 else 0
                height_pad = 2 * (nrows + 1) if nrows > 1 or ncols>1 else 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % folder),fourcc,
                                      20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))


                count = 0
                frame_sub_count = 30
                cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, 6) * 2)
                for k,item in enumerate(img_list):
                    if not item.endswith('g') or os.path.isdir(os.path.join(folder_img,item)): #or (k>=180 and k<290):
                        continue

                    img_path = os.path.join(folder_img, item)
                    save_path = os.path.join(folder_img, item)
                    seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()

                    result = []
                    seg_label = id_remap(F.interpolate(seg_label, size=(resolution_vis, resolution_vis), mode='nearest') * 255)
                    # result.append(vis_condition_img(seg_label))
                    seg_label = seg_label.cuda()

                    half_fp = 240
                    mixstyle = 0
                    n_sample = ncols * nrows
                    if count==0:
                        style_sample = n_sample
                        if not args.miou_filter:
                            styles = mixing_noise(style_sample, args.latent, 0.0, device, unbine=False)
                            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                            styles = torch.cat(styles, dim=0)
                            w_latent = generator.style_map([styles], to_w_space=False)
                        else:
                            style_seg_labels = sample_segmap_from_list(folder_img, img_list[:half_fp+1],samples=4)
                            w_latent = sample_styles_with_miou(style_seg_labels,style_sample, mixstyle=mixstyle, truncation=args.truncation, batch_size=args.batch_size)
                        print('==> sample styles done.')
                    elif count == half_fp:
                        style_sample = (len(img_list)-half_fp)// frame_sub_count
                        style_seg_labels = sample_segmap_from_list(folder_img, img_list[half_fp:], samples=6)
                        w_latent_new = sample_styles_with_miou(style_seg_labels, style_sample, mixstyle=mixstyle,truncation=args.truncation, batch_size=args.batch_size)
                        w_latent = torch.cat((w_latent,w_latent_new,w_latent),dim=0)
                        print('==> sample styles done.')


                    batch_sizes = divide_chunks(list(range(n_sample)),args.batch_size)

                    if count > half_fp:
                        style_idx = (count - half_fp) // frame_sub_count
                        step = (count - half_fp)%frame_sub_count-frame_sub_count//2
                        weight = norm.cdf(step, 0, 6) * cdf_scale
                        w_latent_forward = weight * w_latent[[style_idx+1]] + (1.0 - weight) * w_latent[[style_idx]]
                    else:
                        w_latent_forward = w_latent[[0]]

                    style_masks = None
                    for batch in batch_sizes:
                        fake_img, _, _, _ = generator(w_latent_forward, return_latents=False, condition_img=seg_label.repeat(len(batch),1,1,1), \
                                                      input_is_latent=True, noise=noise, style_mask=style_masks)
                        fake_img = fake_img.clamp(-1.0, 1.0)
                        fake_img = F.interpolate(fake_img, size=(resolution_vis, resolution_vis), mode='nearest')
                        result.append(fake_img.detach().cpu())

                    result = torch.cat(result, dim=0)
                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                    result = (utils.make_grid(result, nrow=ncols) + 1) / 2 * 255
                    out.write((result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0)))
                    count += 1

    # modify style by layer
    elif 11 == args.MODE:
        resolution_vis = 512
        nrows,ncols = 1,4
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            # visList = os.listdir(os.path.join(args.input,'vis'))
            for i, item in enumerate(List):
                if not item.endswith('g') or os.path.isdir(os.path.join(args.input, item)):  # or (k>=180 and k<290):
                    continue


                width_pad = 2 * (ncols + 1) if ncols > 1 or nrows>1 else 0
                height_pad = 2 * (nrows + 1) if nrows > 1 or ncols>1 else 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % item[:-4]), fourcc,
                                      15, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))


                img_path = os.path.join(args.input, item)
                save_path = os.path.join(args.input, item)
                seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()

                result = []
                seg_label = id_remap(F.interpolate(seg_label, size=(resolution_vis, resolution_vis), mode='nearest') * 255)
                # result.append(vis_condition_img(seg_label))
                seg_label = seg_label.cuda()


                mixstyle = 0
                if not args.miou_filter:
                    styles = mixing_noise(2, args.latent, 0.0, device, unbine=False)
                    styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                    styles = torch.cat(styles, dim=0)
                    w_latent = generator.style_map([styles], to_w_space=False)
                else:
                    w_latent = sample_styles_with_miou(seg_label,2, mixstyle=mixstyle, truncation=args.truncation, batch_size=args.batch_size)
                print('==> sample styles done.')

                n_sample = ncols*nrows
                batch_sizes = divide_chunks(list(range(n_sample)),args.batch_size)

                frame_sub_count,std = 40,20
                cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, std) * 2)
                for frame_sub in range(-frame_sub_count // 2, frame_sub_count // 2 + 1):
                    weight = (norm.cdf(frame_sub, 0, std) - norm.cdf(-frame_sub_count // 2, 0, std)) * cdf_scale
                    w_latent_mixed = w_latent[None].repeat(4, 1, 1, 1).clone()

                    # layer style
                    w_latent_mixed[0,0,:4] = w_latent_mixed[0,0,:4]*(1-weight) + w_latent_mixed[0,1,:4]*weight
                    w_latent_mixed[1, 0, 4:10] = w_latent_mixed[1, 0, 4:10] * (1 - weight) + w_latent_mixed[1, 1, 4:10] *  weight
                    w_latent_mixed[2, 0, 10:18] = w_latent_mixed[2, 0, 10:18] * (1 - weight) + w_latent_mixed[2, 1, 10:18] *  weight
                    w_latent_mixed[3] = w_latent_mixed[2, 0] * (1 - weight) + w_latent_mixed[2, 1] *  weight
                    w_latent_mixed = w_latent_mixed[:,0]


                    fake_img, _, _, _ = generator(w_latent_mixed, return_latents=False, condition_img=seg_label.repeat(w_latent_mixed.shape[0],1,1,1), \
                                                  input_is_latent=True, noise=noise)
                    fake_img = fake_img.clamp(-1.0, 1.0)
                    fake_img = F.interpolate(fake_img, size=(resolution_vis, resolution_vis), mode='nearest')


                    fake_img = fake_img.detach().cpu()
                    fake_img = F.interpolate(fake_img, (resolution_vis, resolution_vis))
                    fake_img = (utils.make_grid(fake_img, nrow=ncols) + 1) / 2 * 255
                    out.write((fake_img.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0)))

    # modify style by layer
    elif 12 == args.MODE:
        import lmdb
        from tqdm import tqdm
        from io import BytesIO
        resolution_vis = 256
        nrows,ncols = 16,16
        noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

        styleList = lmdb.open(
            args.input,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


        n_samples = 1
        styles_latent,features = [],[]
        pbar = range(n_samples)
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

        with torch.no_grad():
            for i in pbar:
                pbar.set_description(
                    (
                        'Processing % d of % d'%(i, n_samples)
                    )
                )

                index_seg = 16#np.random.randint(0,25000)
                key = f'{str(index_seg).zfill(5)}'.encode('utf-8')
                with styleList.begin(write=False) as txn:
                    condition_bytes = txn.get(key)
                    buffer = BytesIO(condition_bytes)
                    seg_label = Image.open(buffer)
                    seg_label.save('/home/anpei/code/softgan_test/result/segmap.png')

                    seg_label = transform(seg_label).unsqueeze(0).cuda()
                    seg_label = id_remap(id_raw_to_new(F.interpolate(seg_label, size=(256, 256), mode='nearest')*255)).cuda()

                    length_seg = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

                    styles_count = ncols * nrows
                    mixstyle = 1.0
                    if not args.miou_filter:
                        styles = mixing_noise(styles_count, args.latent, mixstyle, device, unbine=False)
                        styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                        styles = torch.cat(styles, dim=0)
                        w_latent = generator.style_map([styles], to_w_space=False)
                    else:
                        w_latent = sample_styles_with_miou(seg_label, styles_count, mixstyle=0.0,
                                                           truncation=args.truncation, batch_size=args.batch_size)


                    batch_sizes = divide_chunks(list(range(styles_count)), args.batch_size)

                    style_masks = None
                    for batch in batch_sizes:
                        styles_latent.append(w_latent[batch].to('cpu'))
                        fake_img, _, _, feature = generator(w_latent[batch], return_latents=False,
                                                      condition_img=seg_label.repeat(len(batch), 1, 1, 1), \
                                                      input_is_latent=True, noise=noise, style_mask=style_masks)
                        fake_img = fake_img.clamp(-1.0, 1.0).detach().cpu()


                        fake_img = F.interpolate(fake_img, (resolution_vis, resolution_vis))
                        features.append(feature)
                        for k,img in enumerate(fake_img):
                            save_path = os.path.join(os.path.join(args.output, '%06d_%06d.jpg' % (i,batch[k])))
                            utils.save_image(
                                fake_img[[k]],
                                save_path,
                                nrow=1,
                                normalize=True,
                                range=(-1, 1),
                            )

        styles_latent = torch.cat(styles_latent, 0)
        features = torch.cat(features, 0)
        torch.save({'styles_latent':styles_latent,'features':features},'latentCode3.pth')


    # drawing
    elif 13 == args.MODE:
        resolution_vis = 2048
        nrows,ncols = 3,3
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            # visList = os.listdir(os.path.join(args.input,'vis'))
            for i, folder in enumerate(List):
                if folder=='vis'  :#or (folder+'.gif') not in visList
                    continue

                print('Processing folder: %s of %d/%d' % (folder, i, len(List)))
                folder_img = os.path.join(args.input,folder)#,'seg'
                img_list = sorted(os.listdir(folder_img))

                width_pad = 0
                height_pad = 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % folder), fourcc,
                                      20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))


                count = 0
                for k,item in enumerate(img_list):
                    if not item.endswith('g') or os.path.isdir(os.path.join(folder_img,item)):
                        continue


                    img_path = os.path.join(folder_img, item)
                    save_path = os.path.join(folder_img, item)
                    seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()

                    result = []
                    seg_label = id_remap(F.interpolate(seg_label, size=(resolution_vis, resolution_vis), mode='nearest') * 255)
                    # result.append(vis_condition_img(seg_label))
                    seg_label = seg_label.cuda()

                    if count==0:
                        mixstyle = 0
                        if not args.miou_filter:
                            styles = mixing_noise(nrows * ncols, args.latent, 0.0, device, unbine=False)
                            styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                            styles = torch.cat(styles, dim=0)
                            w_latent = generator.style_map([styles], to_w_space=False)
                        else:
                            w_latent = sample_styles_with_miou(seg_label,nrows * ncols, mixstyle=mixstyle, truncation=args.truncation, batch_size=args.batch_size)
                        print('==> sample styles done.')

                    n_sample = ncols*nrows
                    batch_sizes = divide_chunks(list(range(n_sample)),args.batch_size)

                    style_masks = None
                    for batch in batch_sizes:
                        fake_img, _, _, _ = generator(w_latent[batch], return_latents=False, condition_img=seg_label.repeat(len(batch),1,1,1), \
                                                      input_is_latent=True, noise=noise, style_mask=style_masks)
                        fake_img = fake_img.clamp(-1.0, 1.0)
                        fake_img = F.interpolate(fake_img, size=(resolution_vis, resolution_vis), mode='nearest')
                        result.append(fake_img.detach().cpu())

                    if k>0 and int(item[:-4]) - framePre>1:
                        for l in range(int(item[:-4]) - framePre - 1):
                            out.write(resultPre)

                    result = torch.cat(result, dim=0)
                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                    result = (utils.make_grid(result, nrow=ncols, padding=0) + 1) / 2 * 255
                    result = (result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0))
                    # cv2.imwrite(args.output+'%s_%02d.png'%(folder,k),(result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0)))
                    out.write(result)

                    framePre = int(item[:-4])
                    resultPre = result
                    count += 1

    # new video style
    elif 14 == args.MODE:
        resolution_vis = 512
        from tqdm import tqdm
        with torch.no_grad():
            args.nrows = 2
            seg_label_rgb, seg_labels = [], []

            styles_count = 7
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

            pbar = range(100)
            pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
            for g in pbar:
                w_latents = []
                for i, folder in enumerate(sorted(List)):

                    files = os.listdir(os.path.join(args.input, folder))
                    item = files[np.random.randint(0,len(files))]
                    if not item.endswith('g'):
                        continue

                    img_path = os.path.join(args.input, folder,item)
                    save_path = os.path.join(args.output, folder,item)
                    seg_label = Image.open(img_path)
                    seg_label = transform(seg_label).unsqueeze(0).cuda()

                    seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255)
                    seg_labels.append(seg_label)

                    mixstyle = 1.0
                    if not args.miou_filter:
                        styles = mixing_noise(styles_count, args.latent, 1.0, device, unbine=False)
                        styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                        styles = torch.cat(styles, dim=0)
                        w_latent = generator.style_map([styles], to_w_space=False)
                    else:
                        w_latent = sample_styles_with_miou(seg_label, styles_count, mixstyle=0.0,
                                                           truncation=args.truncation, batch_size=args.batch_size,descending=True)

                    w_latents.append(w_latent)
                    nrows, ncols = 1, 4
                    total_count = nrows * ncols


                width_pad =  0
                height_pad = 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%03d.mp4' % g), fourcc, 20, \
                                      (resolution_vis * ncols + width_pad, 2*resolution_vis * nrows + height_pad))

                w_latents = torch.stack(w_latents,dim=1)


                style_masks = []
                for k,seg_label in enumerate(seg_labels):
                    style_masks.append(scatter_to_mask(seg_label, 2, add_flip=False, region=[k]))
                for k,seg_label in enumerate(seg_labels):
                    style_masks.append(scatter_to_mask(seg_label, 2, add_flip=False, region=[k+len(seg_labels)]))
                style_masks = torch.cat(style_masks,dim=0)
                seg_labels = torch.cat(seg_labels,dim=0)

                for frame in range(styles_count-1):

                    frame_sub_count = 40
                    cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, 6) * 2)
                    for frame_sub in range(-frame_sub_count // 2, frame_sub_count // 2 + 1):
                        result = []
                        weight = (norm.cdf(frame_sub, 0, 6) - norm.cdf(-frame_sub_count // 2, 0, 6)) * cdf_scale
                        if 0==frame:
                            w_latent_current = (1.0 - weight) * w_latents[[0]] + weight * w_latents[[frame],[0]].repeat(1,w_latents.shape[1],1,1)
                        else:
                            w_latent_current = (1.0 - weight) * w_latents[[frame-1], [0]] + weight * w_latents[[frame], [0]].repeat(1, w_latents.shape[1], 1, 1)
                        w_latent_current = torch.cat((w_latents[[0]],w_latent_current),dim=0)

                        n_sample = total_count
                        batch_sizes = divide_chunks(list(range(n_sample))*2, 1)
                        batch_sizes_shifted = divide_chunks(list(range(n_sample*2)), 1)

                        # first row
                        for batch,batch_shift in zip(batch_sizes,batch_sizes_shifted):
                            w_latent_current_in = w_latent_current[:,batch].view(-1, 18, 512)
                            fake_img, _, _, _ = generator(w_latent_current_in, return_latents=False,
                                                          condition_img=seg_labels[batch], \
                                                          input_is_latent=True, noise=noise,
                                                          style_mask=style_masks[batch_shift])
                            result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))


                        result = torch.cat(result, dim=0)
                        result = F.interpolate(result, (resolution_vis, resolution_vis))
                        result = (utils.make_grid(result, nrow=ncols, padding=0) + 1) / 2 * 255
                        result = (result.detach().numpy()[[2, 1, 0]]).transpose((1, 2, 0))
                        img = np.zeros(result.shape)
                        img[:] = result
                        # result = 0.6*cv2.putText(img, 'Global Styles', org=(img.shape[1]//40, img.shape[0]//20),
                        #                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #                            fontScale=1.2, color=(255, 255, 255), thickness=img.shape[0]//400) + 0.4 * result
                        out.write(result.astype('uint8'))

                torch.save(w_latents[:,:,0,:].cpu(),os.path.join(args.output, '%03d.th' % g))
                seg_labels = []


    # new video style
    elif 15 == args.MODE:
        resolution_vis = 512
        from tqdm import tqdm
        with torch.no_grad():
            args.nrows = 2
            seg_label_rgb, seg_labels = [], []

            styles_count = 7
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

            pbar = range(100)
            pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
            for g in pbar:
                w_latents = []
                for i, folder in enumerate(sorted(List)):

                    files = os.listdir(os.path.join(args.input, folder))
                    item = files[np.random.randint(0,len(files))]
                    if not item.endswith('g'):
                        continue

                    img_path = os.path.join(args.input, folder,item)
                    save_path = os.path.join(args.output, folder,item)
                    seg_label = Image.open(img_path)
                    seg_label = transform(seg_label).unsqueeze(0).cuda()

                    seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255)
                    seg_labels.append(seg_label)

                    mixstyle = 1.0
                    if not args.miou_filter:
                        styles = mixing_noise(22, args.latent, 1.0, device, unbine=False)
                        styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
                        styles = torch.cat(styles, dim=0)
                        w_latent = generator.style_map([styles], to_w_space=False)
                    else:
                        w_latent = sample_styles_with_miou(seg_label, 22, mixstyle=0.0,
                                                           truncation=args.truncation, batch_size=args.batch_size,descending=True)

                    w_latents.append(w_latent)
                    nrows, ncols = 1, 1
                    total_count = nrows * ncols


                width_pad =  0
                height_pad = 0
                # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                # out = cv2.VideoWriter(os.path.join(args.output, '%03d.mp4' % g), fourcc, 20, \
                #                       (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))

                w_latents = torch.stack(w_latents,dim=1)


                style_masks = []
                for k,seg_label in enumerate(seg_labels):
                    style_masks.append(scatter_to_mask(seg_label, 1, add_flip=False, add_whole=False))

                style_masks = torch.stack(style_masks,dim=0)
                seg_labels = torch.cat(seg_labels,dim=0)

                w_latent_nexts = []
                for n in range(1):
                    regions = [0] + list(range(1,1+20*(n+1))) #+ [0]


                    for frame in range(1,len(regions)):

                        if 0 == regions[frame - 1]:
                            w_latent_last, w_latent_next = w_latents[[0]], w_latents[[frame], [np.random.randint(0,1)]]
                        elif 0 == regions[frame]:
                            w_latent_last, w_latent_next = w_latent_next.clone(), w_latents[[0]]
                        else:
                            w_latent_last = w_latent_next.clone()
                            w_latent_next = w_latents[[frame], [np.random.randint(0,1)]].clone()

                        w_latent_nexts.append(w_latent_next[:,0])
                        frame_sub_count = 40 if n<4 else 30
                        cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, 6) * 2)
                        # for frame_sub in range(-frame_sub_count // 2, frame_sub_count // 2 + 1):
                        #
                        #     weight = (norm.cdf(frame_sub, 0, 6) - norm.cdf(-frame_sub_count // 2, 0, 6)) * cdf_scale
                        for frame_sub in range(1):

                            weight = 1.0
                            w_latent_current = (1.0 - weight) * w_latent_last + weight * w_latent_next
                            w_latent_current = w_latent_current.expand(w_latents[[0]].shape)
                            w_latent_current = torch.cat((w_latents[[0]],w_latent_current),dim=0)

                            n_sample = total_count
                            batch_sizes = divide_chunks(list(range(n_sample)), 1)

                            # first row
                            result = []
                            for batch in batch_sizes:
                                w_latent_current_in = w_latent_current[:,batch].view(-1, 18, 512)
                                fake_img, _, _, _ = generator(w_latent_current_in, return_latents=False,
                                                              condition_img=seg_labels[batch], \
                                                              input_is_latent=True, noise=noise,
                                                              style_mask=style_masks[batch,n])
                                result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))


                            result = torch.cat(result, dim=0)
                            result = F.interpolate(result, (resolution_vis, resolution_vis))
                            result = (utils.make_grid(result, nrow=ncols, padding=0) + 1) / 2 * 255
                            result = (result.detach().numpy()[[2, 1, 0]]).transpose((1, 2, 0))
                            img = np.zeros(result.shape)
                            img[:] = result
                            result = 0.6*cv2.putText(img, 'all_%d'%frame, org=(img.shape[1]//40, img.shape[0]//20),
                                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                       fontScale=1.2, color=(255, 255, 255), thickness=img.shape[0]//400) + 0.4 * result
                            # out.write(result.astype('uint8'))
                            cv2.imwrite(os.path.join(args.output, '%03d_%03d.jpg' % (g,frame)),result.astype('uint8'))


                torch.save(torch.stack(w_latent_nexts,dim=0),os.path.join(args.output, '%03d.th' % g))
                seg_labels = []

    # new video style
    elif 16 == args.MODE:
        resolution_vis = 512
        from tqdm import tqdm
        with torch.no_grad():
            args.nrows = 2
            seg_label_rgb, seg_labels = [], []

            styles_count = 7
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]

            styles = torch.load('styles.th')
            source = styles['source'].view(1, 4, 1, 512).repeat(1, 1, 18, 1)

            pbar = range(100)
            pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
            for g in pbar:
                w_latents = []
                for i, folder in enumerate(sorted(List)):

                    files = os.listdir(os.path.join(args.input, folder))
                    item = files[np.random.randint(0,len(files))]
                    if not item.endswith('g'):
                        continue

                    img_path = os.path.join(args.input, folder,item)
                    save_path = os.path.join(args.output, folder,item)
                    seg_label = Image.open(img_path)
                    seg_label = transform(seg_label).unsqueeze(0).cuda()

                    seg_label = id_remap(F.interpolate(seg_label, size=(args.resolution, args.resolution), mode='nearest') * 255)
                    seg_labels.append(seg_label)


                    nrows, ncols = 1, 4
                    total_count = nrows * ncols


                width_pad =  0
                height_pad = 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%03d.mp4' % g), fourcc, 20, \
                                      (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))



                style_masks = []
                for k,seg_label in enumerate(seg_labels):
                    style_masks.append(scatter_to_mask(seg_label, 6, add_flip=False, add_whole=False))

                style_masks = torch.stack(style_masks,dim=0)
                seg_labels = torch.cat(seg_labels,dim=0)

                w_latent_nexts = []
                for n in range(6):
                    next_list = np.random.permutation(styles[groupName[n]].shape[0])[:4]
                    w_latent_nexts.append(next_list)
                    regions = [0] + list(next_list) + [0]

                    for j,frame in enumerate(range(1,len(regions))):

                        if 0 == j:
                            w_latent_last, w_latent_next = source, styles[groupName[n]][[regions[frame]]]
                        elif len(regions)-2 == j:
                            w_latent_last, w_latent_next = w_latent_next.clone(), source
                        else:
                            w_latent_last = w_latent_next.clone()
                            w_latent_next = styles[groupName[n]][[regions[frame]]]

                        frame_sub_count = 40 if n<4 else 30
                        cdf_scale = 1.0 / (1.0 - norm.cdf(-frame_sub_count // 2, 0, 6) * 2)
                        for frame_sub in range(-frame_sub_count // 2, frame_sub_count // 2 + 1):

                            weight = (norm.cdf(frame_sub, 0, 6) - norm.cdf(-frame_sub_count // 2, 0, 6)) * cdf_scale

                            w_latent_current = (1.0 - weight) * w_latent_last + weight * w_latent_next
                            w_latent_current = w_latent_current.expand(source.shape)
                            w_latent_current = torch.cat((source,w_latent_current),dim=0)

                            n_sample = total_count
                            batch_sizes = divide_chunks(list(range(n_sample)), 1)

                            # first row
                            result = []
                            for batch in batch_sizes:
                                w_latent_current_in = w_latent_current[:,batch].view(-1, 18, 512)
                                fake_img, _, _, _ = generator(w_latent_current_in, return_latents=False,
                                                              condition_img=seg_labels[batch], \
                                                              input_is_latent=True, noise=noise,
                                                              style_mask=style_masks[batch,n])
                                result.append(fake_img.detach().cpu().clamp(-1.0, 1.0))


                            result = torch.cat(result, dim=0)
                            result = F.interpolate(result, (resolution_vis, resolution_vis))
                            result = (utils.make_grid(result, nrow=ncols, padding=0) + 1) / 2 * 255
                            result = (result.detach().numpy()[[2, 1, 0]]).transpose((1, 2, 0))
                            img = np.zeros(result.shape)
                            img[:] = result
                            # result = 0.6*cv2.putText(img, 'all_%d'%frame, org=(img.shape[1]//40, img.shape[0]//20),
                            #                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            #                            fontScale=1.2, color=(255, 255, 255), thickness=img.shape[0]//400) + 0.4 * result
                            out.write(result.astype('uint8'))
                            # cv2.imwrite(os.path.join(args.output, '%03d_%03d.jpg' % (g,frame)),result.astype('uint8'))


                torch.save(torch.from_numpy(np.stack(w_latent_nexts,axis=0)),os.path.join(args.output, '%03d.th' % g))
                seg_labels = []

    # given styles
    elif 17 == args.MODE:
        resolution_vis = 512
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            # visList = os.listdir(os.path.join(args.input,'vis'))

            stylesIdx = torch.load('011.th')
            styles = torch.load('styles.th')

            fileOrder = np.random.permutation(len(List)).astype('int')
            w_latents = []
            for i in range(6):
                w_latents.append(styles[groupName[i]][stylesIdx[i]])
            w_latents = torch.cat(w_latents,dim=0)
            for i,w_latent in enumerate(w_latents):
                w_latent = w_latent.clone().view(1,1,512).repeat(1,18,1)
                for j in range(5):
                    item = List[fileOrder[i*5+j]]

                    img_path = os.path.join(args.input, item)
                    seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()

                    seg_label = id_remap(
                        F.interpolate(seg_label, size=(resolution_vis, resolution_vis), mode='nearest') * 255)
                    seg_label = seg_label.cuda()


                    style_masks = None
                    fake_img, _, _, _ = generator(w_latent, return_latents=False,
                                                  condition_img=seg_label, \
                                                  input_is_latent=True, noise=noise, style_mask=style_masks)
                    fake_img = fake_img.clamp(-1.0, 1.0)
                    fake_img = F.interpolate(fake_img, size=(resolution_vis, resolution_vis), mode='nearest')
                    result = fake_img.detach().cpu()

                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                    result = (utils.make_grid(result, nrow=1) + 1) / 2 * 255
                    cv2.imwrite(args.output+'%02d_%02d.png'%(i,j),(result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0)))
                    count += 1

    # drawing given style
    elif 18 == args.MODE:
        resolution_vis = 1024
        with torch.no_grad():
            seg_label_rgb = []
            result, count = [], 0
            noise = [getattr(generator.noises, f'noise_{i}') for i in range(generator.num_layers)]
            styles = torch.load('harryStyle.th')
            nrows, ncols = 1,styles.shape[0]
            for i, folder in enumerate(List):

                folder_img = os.path.join(args.input,folder)#,'seg'
                img_list = sorted(os.listdir(folder_img))

                width_pad = 0
                height_pad = 0
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(os.path.join(args.output, '%s.mp4' % folder), fourcc,
                                      20, (resolution_vis * ncols + width_pad, resolution_vis * nrows + height_pad))


                count = 0
                for k,item in enumerate(img_list):
                    if not item.endswith('g') or os.path.isdir(os.path.join(folder_img,item)):
                        continue


                    img_path = os.path.join(folder_img, item)
                    save_path = os.path.join(folder_img, item)
                    seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()

                    result = []
                    seg_label = id_remap(F.interpolate(seg_label, size=(resolution_vis, resolution_vis), mode='nearest') * 255)
                    # result.append(vis_condition_img(seg_label))
                    seg_label = seg_label.cuda()

                    if count==0:
                        mixstyle = 0
                        if not args.miou_filter:
                            w_latent = styles.view(-1,1,512).repeat(1,18,1).cuda()
                        print('==> sample styles done.')

                    n_sample = ncols*nrows
                    batch_sizes = divide_chunks(list(range(n_sample)),args.batch_size)

                    style_masks = None
                    for batch in batch_sizes:
                        fake_img, _, _, _ = generator(w_latent[batch], return_latents=False, condition_img=seg_label.repeat(len(batch),1,1,1), \
                                                      input_is_latent=True, noise=noise, style_mask=style_masks)
                        fake_img = fake_img.clamp(-1.0, 1.0)
                        fake_img = F.interpolate(fake_img, size=(resolution_vis, resolution_vis), mode='nearest')
                        result.append(fake_img.detach().cpu())

                    if k>0 and int(item[:-4]) - framePre>1:
                        for l in range(int(item[:-4]) - framePre - 1):
                            out.write(resultPre)

                    result = torch.cat(result, dim=0)
                    result = F.interpolate(result, (resolution_vis, resolution_vis))
                    result = (utils.make_grid(result, nrow=ncols, padding=0) + 1) / 2 * 255
                    result = (result.numpy().astype('uint8')[[2,1,0]]).transpose((1,2,0))
                    out.write(result)

                    framePre = int(item[:-4])
                    resultPre = result
                    count += 1
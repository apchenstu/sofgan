'''
# example CUDA_VISIBLE_DEVICES=0 python projector.py -i /root/anpei/code/styleGAN3/dataset/back_project_test/ -o /root/anpei/code/styleGAN3/result/back_project-w --ckpt /root/anpei/code/styleGAN3/checkpoint/tf-1024-conv-seg-less5-7/279999.pt --step 1000 --w_plus
'''
import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

import lpips, cv2
from modules.model_seg_input2 import Generator2
from modules.BiSeNet import BiSeNet

import numpy as np
remap_list = torch.tensor([0,1,2,2,3,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16]).float()
def id_remap(seg):
    #['background'0,'skin'1, 'l_brow'2, 'r_brow'3, 'l_eye'4, 'r_eye'5,'r_nose'6, 'l_nose'7, 'mouth'8, 'u_lip'9,
    # 'l_lip'10, 'l_ear'11, 'r_ear'12, 'ear_r'13, 'eye_g'14, 'neck'15, 'neck_l'16, 'cloth'17, 'hair'18, 'hat'19]
    return remap_list[seg.long()]

transform_seg = transforms.Compose(
    [
        # transforms.RandomAffine(20,translate=(0.2,0.2)),
        transforms.ToTensor(),
    ]
)

def vis_condition_img(img):
    part_colors = [[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],#'skin',1 'eye_brow'2,  'eye'3
                    [240, 157, 240], [255, 212, 255], #'r_nose'4, 'l_nose'5
                    [31, 162, 230], [127, 255, 255], [127, 255, 255],#'mouth'6, 'u_lip'7,'l_lip'8
                    [0, 255, 85], [0, 255, 170], #'ear'9 'ear_r'10
                    [255, 255, 170],
                    [127, 170, 255], [85, 0, 255], [255, 170, 127], #'neck'11, 'neck_l'12, 'cloth'13
                    [212, 127, 255], [0, 170, 255],#, 'hair'14, 'hat'15
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]
    H,W = img.shape
    condition_img_color = np.zeros((H,W,3))

    num_of_class = int(np.max(img))
    for pi in range(1, num_of_class + 1):
        index = np.where(img == pi)
        condition_img_color[index[0], index[1],:] = part_colors[pi]
    return condition_img_color

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) \
                   + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)

            if size <= 8:
                break

            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255) \
        .type(torch.uint8).permute(0, 2, 3, 1).to('cpu').numpy()


def absoluteFilePaths(directory):
    path = []
    files = sorted(os.listdir(directory))
    for item in files:
        if item.endswith('g'):
            path.append(os.path.join(os.path.abspath(directory), item))
    return path


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--mse', type=float, default=0)
    parser.add_argument('--w_plus', action='store_true')
    parser.add_argument('--m_plus', action='store_true')
    parser.add_argument('--batch', type=int,default=1)
    parser.add_argument('-i', '--files', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    args.channel_multiplier = 2
    args.condition_dim = 0
    args.condition_path = args.files

    n_mean_latent = 10000

    resize = min(args.resolution, 1024)

    transform = transforms.Compose([transforms.Resize(resize),
                                    transforms.CenterCrop(resize),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])

    imgs, segmaps, files = [], [],[]
    if os.path.isdir(args.files):
        files = absoluteFilePaths(args.files)
    else:
        files = [args.files]
    # print(args.files)


    # segmentaion
    segNet = None
    n_classes = 20
    segNet = BiSeNet(n_classes=n_classes)

    checkpoint = torch.load('modules/segNet-20Class.pth')
    segNet.load_state_dict(checkpoint)
    segNet = segNet.to(device)
    segNet.eval()

    # generator
    g_ema = Generator2(args).to(device)
    g_ema.eval()
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=device.startswith('cuda'))

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        # pre calc std and bias

    input_files = []
    for i,imgfile in enumerate(files):

        if not imgfile.endswith('g'):
            continue


        imgfile_path = os.path.abspath(imgfile)
        img = transform(Image.open(imgfile_path).convert('RGB'))
        imgs.append(img)
        input_files.append(imgfile_path)
        if os.path.exists(imgfile[:-4]+f'_seg.png'):
            segmaps.append(torch.from_numpy(np.array(Image.open(imgfile[:-4]+f'_seg.png'))))

        if len(imgs)==args.batch or i==len(args.files)-1:

            imgs = torch.stack(imgs, dim=0).cuda()
            if len(segmaps):
                input_seg = id_remap(torch.stack(segmaps)).cuda()
            else:
                with torch.no_grad():
                    input_size = 512
                    input_seg = torch.nn.functional.interpolate(imgs, size=(512, 512),
                                                                mode='bilinear', align_corners=True)
                    input_seg = segNet(input_seg)[0].detach()
                    input_seg = torch.nn.functional.interpolate(input_seg, size=(input_size, input_size),
                                                                mode='bilinear', align_corners=True)
                    input_seg = id_remap(input_seg.argmax(1, keepdim=True)).cuda()
                    for j, item in enumerate(input_seg):
                        img = item[0].cpu().numpy().astype('uint8')
                        cv2.imwrite(os.path.join(args.output, os.path.basename(imgfile)[:-4]+'_seg.png'),img)

            n_classes = 17
            latent_in = latent_mean.detach().clone()[None, None].repeat(args.batch,1,1)
            if args.w_plus:
                latent_in = latent_in.repeat(1, g_ema.n_latent, 1)
            if args.m_plus:
                class_probability = torch.randn(n_classes, device=device, requires_grad=True)
                latent_in = latent_in.repeat(2, 1, 1)
            latent_in.requires_grad = True

            noises = g_ema.make_noise()
            for noise in noises:
                noise.requires_grad = True

            parameters = [latent_in] + noises
            if args.m_plus:
                parameters += [class_probability]
            optimizer = optim.Adam(parameters, lr=args.lr)

            pbar = tqdm(range(args.step))
            latent_path = []

            for k in pbar:
                t = k / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]['lr'] = lr
                noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                latent_n = latent_noise(latent_in, noise_strength.item())
                # if not args.w_plus:
                #     latent_n = [latent_n]


                style_mask = None
                if args.m_plus:
                    style_mask = class_probability[input_seg.long()]
                    style_mask = torch.cat((style_mask,1.0-style_mask),1)

                img_gen, _, _, _ = g_ema(latent_n, condition_img=input_seg, noise=noises, input_is_latent=args.w_plus, style_mask=style_mask)


                batch, channel, height, width = img_gen.shape

                # if height > 256:
                #     factor = height // 256
                #
                #     img_gen = img_gen.reshape(
                #         batch, channel, height // factor, factor, width // factor, factor
                #     )
                #     img_gen = img_gen.mean([3, 5])

                p_loss = percept(img_gen, imgs).sum()
                n_loss = noise_regularize(noises)
                mse_loss = F.mse_loss(img_gen, imgs)

                loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss
                # loss = p_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                noise_normalize_(noises)
                # print(class_probability)
                # if (k + 1) % 100 == 0:
                #     latent_path.append(latent_in.detach().clone())

            pbar.set_description((f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
                                  f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'))


            with torch.no_grad():
                img_gen, _, _, _ = g_ema(latent_n, condition_img=input_seg, noise=noises, input_is_latent=args.w_plus)

            if not os.path.exists(args.output):
                os.mkdir(args.output)
            img_ar = make_image(img_gen)
            for j, input_name in enumerate(input_files):
                img_name = os.path.join(args.output, os.path.basename(input_name))
                pil_img = Image.fromarray(img_ar[j])
                pil_img.save(img_name)

                result_file = {'noises': noises, 'latent_n': latent_n[j], 'style_mask':style_mask}#, 'class_probability':class_probability
                torch.save(result_file, img_name[:-4]+'.npy')
            imgs,segmaps,input_files = [],[],[]

            # filename = os.path.join(args.output, 'reProject.pt')
            # torch.save(result_file, filename)


            # stylizing the video
            # folder = f'{args.files}/{os.path.basename(imgfile)[:-4]}/'
            # save_root = f'{args.output}/{os.path.basename(imgfile)[:-4]}/'
            # os.makedirs(save_root, exist_ok=True)
            # List = sorted(os.listdir(folder))
            # for j, item in enumerate(List):
            #     img_path = os.path.join(folder, item)
            #     save_path = os.path.join(save_root, item)
            #
            #     seg_label = Image.open(img_path)
            #     seg_label = transform_seg(seg_label).unsqueeze(0)
            #
            #     seg_label = id_remap(F.interpolate(seg_label, size=(256,256), mode='nearest') * 255).to(device)
            #
            #     img_gen, _, _, _ = g_ema(latent_n, condition_img=seg_label, noise=noises, input_is_latent=args.w_plus,style_mask=style_mask)
            #
            #     img_gen = torch.clamp(img_gen, -1, 1).detach().cpu()
            #     utils.save_image(
            #         img_gen,
            #         save_path,
            #         nrow=1,
            #         normalize=True,
            #         range=(-1, 1),
            #         padding = 0
            #     )

            # adjust regional styles
            # input_seg_long = torch.squeeze(input_seg.long())
            # latent_n = torch.cat((latent_mean[None,None].repeat(1,18,1),latent_n),dim=0)
            # style_mask = torch.zeros_like(input_seg_long)
            # style_mask = torch.stack((style_mask, 1.0 - style_mask), 0).float()
            # for s in range(10):
            #     style_mask[0,input_seg_long==15],style_mask[1,input_seg_long==15] = s/9.0, 1.0-s/9.0
            #     style_mask[0, input_seg_long == 14], style_mask[1, input_seg_long == 14] = s / 9.0, 1.0 - s / 9.0
            #     img_gen, _, _, _ = g_ema(latent_n, condition_img=input_seg, noise=noises, input_is_latent=args.w_plus, style_mask=style_mask[None])
            #
            #     img_ar = make_image(img_gen)
            #     img_name = f'{args.output}/{os.path.basename(input_name)[:-4]}_{s:02d}.png'
            #     pil_img = Image.fromarray(img_ar[j])
            #     pil_img.save(img_name)

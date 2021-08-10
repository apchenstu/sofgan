import sys

sys.path.insert(0,'..')


import argparse


from modules.model_seg_input import Generator
from modules.BiSeNet import BiSeNet
from utils import *

device = 'cuda'

def make_noise(batch, styles_dim, style_repeat, latent_dim, n_noise, device):
    noises = torch.randn(n_noise, batch, styles_dim, latent_dim, device=device).repeat(1, 1, style_repeat, 1)
    return noises

def mixing_noise(batch, latent_dim, prob, device, unbine=True):
    n_noise = 1
    style_dim = 2 if random.random() < prob else 1
    style_repeat = 2 // style_dim  # if prob>0 else 1
    styles = make_noise(batch, style_dim, style_repeat, latent_dim, n_noise, device)
    return styles.unbind(0) if unbine else styles

def initFaceParsing(n_classes=20):
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load('../modules/segNet-20Class.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    return net, to_tensor

def parsing_img(bisNet, image, to_tensor, argmax=True):
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0).cuda()
        segmap = bisNet(img)[0]
        if argmax:
            segmap = segmap.argmax(1, keepdim=True)
        segmap = id_remap(segmap)
    return img, segmap


def init_deep_model(ckpt_path, style_path=None):
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

    cmd = f'--ckpt {ckpt_path} \
    --resolution 1024  --truncation 0.7'
    args = parser.parse_args(cmd.split())

    # define networks
    args.latent = 512
    args.n_mlp = 8
    args.condition_path = args.input
    generator = Generator(args).eval().to(device)

    ckpt = torch.load(args.ckpt)
    generator.load_state_dict(ckpt['g_ema'])

    batch_size = 4
    latent_av = cal_av(generator, batch_size, args.latent)

    if style_path is None:
        mixstyle = 0.0
        styles = mixing_noise(36, args.latent, mixstyle, device, unbine=False)
        styles = to_w_style(generator.style_map_norepeat, styles, latent_av, trunc_psi=args.truncation)
        styles = torch.cat(styles, dim=0)
        w_latent = generator.style_map([styles], to_w_space=False)
        w_latent = w_latent.view(-1, 2, w_latent.shape[-2], w_latent.shape[-1])

    del ckpt
    torch.cuda.empty_cache()

    return w_latent.cuda(), generator.cuda().eval()


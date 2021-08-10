import argparse
from utils import *


from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data

from torchvision import transforms, utils
from tqdm import tqdm
from torch.nn import functional as F
from modules.loss import g_path_regularize, d_logistic_loss, d_r1_loss, g_nonsaturating_loss


from dataset import MultiResolutionDataset
from modules.model_seg_input import Discriminator
from modules.model_seg_input import Generator

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    start_iter = args.start_iter // get_world_size() // args.batch
    pbar = range(args.iter // get_world_size() // args.batch)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    seg_loss = torch.tensor(0.0, device=device)
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg, seg_loss_val, shift_loss_val = 0, 0, 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))


    sample_condition_img, sample_conditions, condition_img_color = random_condition_img(args.n_sample)

    if get_rank() == 0:
        os.makedirs(f'sample', exist_ok=True)
        os.makedirs(f'sample/{args.name}', exist_ok=True)
        os.makedirs(f'ckpts/{args.name}', exist_ok=True)
        if args.with_tensorboard:
            os.makedirs(f'tensorboard/{args.name}', exist_ok=True)
            writer = SummaryWriter(f'tensorboard/{args.name}')

    for idx in pbar:
        i = idx + start_iter

        if i > args.iter:
            print('Done!')
            break

        real_img, condition_img = next(loader)
        real_img = real_img.to(device)


        if args.condition_path is not None:
            condition_img = condition_img.to(device)
        else:
            condition_img = None


        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _, _, _ = generator(noise, condition_img=condition_img)
        if args.with_rgbs:
            condition_img_encoder = F.interpolate(condition_img, size=args.resolution, mode='nearest')
            real_img = torch.cat((real_img, condition_img_encoder), dim=1)
        fake_pred, _ = discriminator(fake_img)
        real_pred, real_pred_feat = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _, _,parsing_feature = generator(noise, condition_img=condition_img)

        fake_pred, fake_pred_feat = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        loss_dict['g'] = g_loss

        loss_dict['seg'] = seg_loss
        loss_dict['shift_loss'] = seg_loss

        loss = g_loss
        generator.zero_grad()
        loss.backward()
        g_optim.step()

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        g_regularize = i % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent, args.mixing, device
            )
            if args.condition_path is not None:
                condition_img = condition_img[range(path_batch_size)]
                condition_img.requires_grad = True


            fake_img, latents, _, _ = generator(noise, return_latents=True, condition_img=condition_img)

            path_loss, mean_path_length, path_lengths, isNaN = g_path_regularize(fake_img, latents, mean_path_length)

            generator.zero_grad()
            weighted_path_loss = args.g_reg_every * args.path_regularize  * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
            if not isNaN:
                g_optim.step()
            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if args.condition_path is not None and (0 == i % args.g_reg_every):
            seg_loss_val = loss_reduced['seg'].mean().item()
            shift_loss_val = loss_reduced['shift_loss'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'mean path: {mean_path_length_avg:.4f}'
                )
            )

            if args.with_tensorboard:
                writer.add_scalar('Loss/Generator', g_loss_val, i)
                writer.add_scalar('Loss/Discriminator', d_loss_val, i)
                writer.add_scalar('Loss/R1', r1_val, i)
                writer.add_scalar('Loss/Path Length', path_length_val, i)
                writer.add_scalar('Loss/mean path', mean_path_length_avg, i)


                if args.condition_path is not None:
                    writer.add_scalar('Loss/seg_img', seg_loss_val, i)
                    writer.add_scalar('Loss/shift_loss', shift_loss_val, i)

            steps = get_world_size() * args.batch * (1 + i)
            if steps % 100000 < get_world_size() * args.batch or (steps<1000 and steps%500==get_world_size() * args.batch):
                with torch.no_grad():
                    g_ema.eval()
                    samples,featuresMaps,parsing_features = [],[],[]
                    small_batch = args.n_sample // args.batch
                    if 0 != args.n_sample % args.batch:
                        small_batch += 1

                    # only condition change
                    rows = int(args.n_sample ** 0.5)
                    if args.condition_path is not None:
                        sample_z = mixing_noise(rows, args.latent, args.mixing, device)
                        sample_z = sample_z.unsqueeze(1).repeat(1,rows,1,1).view(args.n_sample,sample_z.shape[1],sample_z.shape[2])
                    else:
                        sample_z = mixing_noise(args.n_sample, args.latent, args.mixing, device)


                    for k in range(small_batch):

                        start,end = k* args.batch,(k+1)* args.batch
                        if k == small_batch-1:
                            end = sample_z.shape[0]

                        if args.condition_path is not None:
                            sample_condition_img_sub = sample_condition_img[start:end]
                            sample_condition_img_sub = random_affine(sample_condition_img_sub.clone(), Scale=0.0).to(device)
                        else:
                            sample_condition_img_sub = None

                        sample, _, _, _ = g_ema(sample_z[start:end], condition_img=sample_condition_img_sub)
                        samples.append(sample.cpu().detach())

                    samples = torch.cat(samples,dim=0)

                    nrow = int(args.n_sample ** 0.5)
                    c,h,w = samples.shape[-3:]
                    samples = samples.reshape(nrow,nrow,c,h,w).transpose(1,0).reshape(-1,c,h,w)
                    utils.save_image(
                        samples,
                        f'sample/{args.name}/{str(steps).zfill(6)}.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    if 0 ==i:
                        c, h, w = condition_img_color.shape[-3:]
                        condition_img_color = condition_img_color.reshape(nrow, nrow, c, h, w).transpose(1, 0).reshape(-1, c, h, w)
                        utils.save_image(
                            condition_img_color,
                            f'sample/{args.name}/seg_vis.png',
                            nrow=nrow,
                            normalize=True,
                            range=(-1, 1),
                        )


            if (steps+get_world_size()*args.batch) % 100000 < get_world_size()*args.batch and steps != args.start_iter:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        # 'g_optim': g_optim.state_dict(),
                        # 'd_optim': d_optim.state_dict(),
                    },
                    f'ckpts/{args.name}/{str(steps).zfill(6)}.pt',
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('--name', type=str, default='pose_condition')
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_sample', type=int, default=25)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--with_rgbs', action='store_true')
    parser.add_argument('--with_tensorboard', action='store_true')
    parser.add_argument('--with_gan_feat_loss', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--lamb_condiction', type=float, default=2)
    parser.add_argument('--condition_path', type=str, default=None)
    parser.add_argument('--apex', action='store_true')

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(args).to(device)
    discriminator = Discriminator(args).to(device)


    segNet = None
    g_ema = Generator(args).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        # style_params,
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(ckpt_name.split('.')[0].split('-')[-1])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'])
        discriminator.load_my_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])


        del ckpt
        torch.cuda.empty_cache()

    percept = None
    if args.with_gan_feat_loss:
        import lpips
        percept = lpips.PerceptualLoss(model='net-lin', net='vgg')

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )


    dataset = MultiResolutionDataset(args.path, transform, args.resolution, condition_path=args.condition_path)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers = args.num_workers,
    )

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)

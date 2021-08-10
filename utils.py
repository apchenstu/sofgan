import torch,os
import random
import numpy as np
from PIL import Image

from torch.utils import data
import torch.distributed as dist
from torchvision import transforms
from torch.nn import functional as F

remap_list_np = np.array([0,1,2,2,3,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16]).astype('float')
def id_remap_np(seg):
    return remap_list_np[seg.astype('int')]

def random_condition_img(batch):
    condition = np.load('./ckpts/val_samples.npy', allow_pickle=True)[()]
    condition_img = id_remap_np(condition['condition_img'])
    condition_img_color = vis_condition_img_np(condition_img[:batch])
    condition_img =  torch.from_numpy(condition_img[:batch]).float()
    condition_img_color = torch.from_numpy(condition_img_color)
    return condition_img, [], condition_img_color

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def random_affine(tensor,Scale=0.5):
    N,C,H,W = tensor.shape
    results = []
    tensor = tensor.byte()
    for i in range(N):
        sample_condition_img = transforms.ToPILImage(mode='L')(tensor[i])
        sample_condition_img = sample_condition_img.resize((int(H * (1.0+Scale)), int(W * (1.0+Scale))),resample=Image.NEAREST)
        sample_condition_img = torch.round(transforms.ToTensor()(sample_condition_img) * 255)
        top_left = [int(np.random.rand() * Scale * H), int(np.random.rand() * Scale * W)]
        sample_condition_img = sample_condition_img[:,top_left[0]:top_left[0]+H,top_left[1]:W+top_left[1]]
        results.append(sample_condition_img)
    return torch.cat(results, dim=0).unsqueeze(1)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def make_noise(batch, styles_dim, style_repeat, latent_dim, device):
    noises = torch.randn(batch, styles_dim, latent_dim, device=device).repeat(1,style_repeat,1)
    return noises


def mixing_noise(batch, latent_dim, prob, device):
    style_dim = 2 if random.random() < prob else 1
    style_repeat = 2//style_dim if prob>0 else 1
    styles = make_noise(batch, style_dim, style_repeat, latent_dim, device)
    return styles


def vis_condition_img_np(img):
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
    N,C,H,W = img.shape
    condition_img_color = np.zeros((N,3,H,W))

    num_of_class = int(np.max(img))
    for pi in range(1, num_of_class + 1):
        index = np.where(img[:,0] == pi)
        condition_img_color[index[0],:,index[1], index[2]] = part_colors[pi]
    condition_img_color = condition_img_color / 255 * 2.0 - 1.0
    return condition_img_color

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

def get_2d_nose(style_img_size, batch=1, c=1):
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
    return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n)]


def gaussion(x, sigma=1, mu=0):
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))



# ['background'0,'skin'1, 'eye_brow'2, 'eye'3,'r_nose'4, 'l_nose'5, 'mouth'6, 'u_lip'7,
# 'l_lip'8, 'ear'9, 'ear_r'10, 'eye_g'11, 'neck'12, 'neck_l'13, 'cloth'14, 'hair'15, 'hat'16]
IDList = [np.arange(17).tolist(), [0], [1, 4, 5, 9, 12], [15], [6, 7, 8, 3], [11, 13, 14, 16, 10]]
groupName = ['Global', 'Background', 'Complexion', 'Hair', 'Eyes & Mouth', 'Wearings']


def scatter_to_mask(segementation, out_num=1, add_whole=True, add_flip=False, region=None):
    segementation = scatter(segementation)
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


def scatter_to_mask_random(segementation, out_num=1, add_whole=True, add_flip=False, idx=None):
    segementation = scatter(segementation)
    masks = []

    if add_whole:
        mask = torch.sum(segementation, dim=1, keepdim=True).clamp(0.0, 1.0)
        masks.append(torch.cat((mask, 1.0 - mask), dim=1))
    if add_flip:
        masks.append(torch.cat((1.0 - mask, mask), dim=1))

    for i in range(out_num - add_whole - add_flip):
        if idx is None:
            idList = IDList[np.random.randint(0, len(IDList))]
        else:
            idList = IDList[idx[i]]
        mask = torch.sum(segementation[:, idList], dim=1, keepdim=True).clamp(0.0, 1.0)
        masks.append(torch.cat((1.0 - mask, mask), dim=1))

    masks = torch.cat(masks, dim=0)
    return masks

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


def scatter(codition_img, source=None, classSeg=20):
    batch, c, height, width = codition_img.size()
    input_label = torch.cuda.BoolTensor(batch, classSeg, height, width).zero_()

    if source is None:
        source = 1
    return input_label.scatter_(1, codition_img.long(), source)


def mIOU(source, target):
    source, target = scatter(source), scatter(target)

    mIOU = torch.mean(torch.div(
        torch.sum(source * target, dim=[2, 3]).float(),
        torch.sum((source + target)>0, dim=[2, 3]).float() + 1e-6), dim=1)
    return mIOU


# def sample_segmap_from_list(folder_img, img_list, samples=4):
#     segmaps = []
#     idx = np.random.randint(0, len(img_list), samples)
#     for item in idx:
#         img_path = os.path.join(folder_img, img_list[item])
#         seg_label = transform(Image.open(img_path)).unsqueeze(0).cuda()
#         seg_label = id_remap(F.interpolate(seg_label, size=(128, 128), mode='nearest') * 255)
#         segmaps.append(seg_label)
#     segmaps = torch.cat(segmaps).cuda()
#     return segmaps

remap_list = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16]).float()
def id_remap(seg):
    return remap_list[seg.long()].to(seg.device)


segClass,final_channel = 17,3
with_classwise_noise = False
def scatter(codition_img,source=None, label_size=(128,128)):
    batch, c, height, width = codition_img.size()

    if height != label_size[0] or width != label_size[1]:
        codition_img= F.interpolate(codition_img, size=label_size, mode='nearest')
        if source is not None:
            source = F.interpolate(source, size=label_size, mode='nearest')

    input_label = torch.cuda.FloatTensor(batch, segClass,label_size[0],label_size[1]).zero_()

    if source is None:
        source = 1.0
    return input_label.scatter_(1, codition_img.long(),source)


classBin = torch.tensor([0,1,2,3,1,1,4,5,5,1,6,7,1,8,9,10,11])
semanticGroups = [[0], [1, 4, 5, 9, 12], [2], [3], [6], [7, 8], [10], [11], [13], [14], [15], [16]]
def scatter_to_mask(segementation, labels):
    masks = []
    for i in range(segementation.shape[0]):
        groups = torch.unique(classBin[labels[i]])
        index = np.arange(len(groups))
        np.random.shuffle(index)
        ind = [j for sub in groups[index[:len(groups)//2]] for j in semanticGroups[sub]]
        segementation_temp = segementation[i,ind].unsqueeze(0)
        mask = (torch.sum(segementation_temp,dim=1,keepdim=True)).clamp(0.0,1.0)#*random.random()
        masks.append(torch.cat((mask,1.0-mask),dim=1))
    masks = torch.cat(masks, dim=0)
    return masks

def scatter_to_mask_perregion(segementation, labels):
    masks = []
    for i in range(segementation.shape[0]):
        groups = torch.unique(classBin[labels[i]])
        index = np.arange(len(groups))
        ind = [j for sub in groups[index[:len(groups) // 2]] for j in semanticGroups[sub]]
        possibility = torch.rand(len(ind),device=segementation.device).view(-1,1,1)
        segementation_temp = (possibility * segementation[i,ind]).unsqueeze(0)
        mask = torch.sum(segementation_temp,dim=1,keepdim=True).clamp(0.0,1.0)#*random.random()
        masks.append(torch.cat((mask,1.0-mask),dim=1))
    masks = torch.cat(masks, dim=0)
    return masks

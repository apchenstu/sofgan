import torch,math
import numpy as np
from scipy import linalg
from torch import autograd,nn
from torch.nn import functional as F
# import cv2, dlib

EPS = 1e-8
def g_path_regularize_our(generator, latents, seg_batch, fake_img, mean_path_length, decay=0.01):
    std = latents.std(dim=0, keepdim=True) + EPS
    w_styles_2 = latents + torch.randn(latents.shape).cuda()*(std + EPS)*1.0
    pl_images, _, _, _ = generator(w_styles_2,  condition_img=seg_batch,input_is_latent=True)
    pl_lengths = ((pl_images - fake_img) ** 2).mean(dim=(1, 2, 3))

    if torch.isnan(pl_lengths.max()):
        pl_lengths = torch.tensor(0.0).cuda()

    path_mean = mean_path_length +  decay * ((pl_lengths.mean() - mean_path_length) ** 2)
    path_penalty = (pl_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), pl_lengths

import time
def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    # grad2, = autograd.grad(
    #     outputs=(fake_img * noise).sum(), inputs=segmap, create_graph=True, allow_unused=True
    # )

    isNaN=False
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    if sum(path_lengths < EPS)>0:
        isNaN = True
        path_lengths = torch.tensor(1.0).cuda()*mean_path_length

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths, isNaN


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def load_param_whitening(args):
    meta = np.load('./modules/param_whitening.npy',allow_pickle=True)[()]
    param_mean, param_std = meta['param_mean'],meta['param_std']
    param_mean, param_std = torch.from_numpy(param_mean).cuda(),torch.from_numpy(param_std).cuda()

    return {'param_mean':param_mean,'param_std':param_std}


def parse_params(params, param_whitening):

    poses,shape,expression = [],[],[]
    for param in params:
        param_decoded = param * param_whitening['param_std'] + param_whitening['param_mean']
        Ps = param_decoded[:12].view(3, -1)  # camera matrix
        s, R, t3d = P2sRt(Ps)

        poses.append(matrix2angle(R))  # yaw, pitch, roll
        shape.append(param[12:52])
        expression.append(param[52:])
    poses,shape,expression = torch.stack(poses,dim=0),torch.stack(shape,dim=0),torch.stack(expression,dim=0)
    return poses,shape,expression

def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (torch.norm(R1) + torch.norm(R2)) / 2.0
    r1 = R1 / torch.norm(R1)
    r2 = R2 / torch.norm(R2)
    r3 = torch.cross(r1, r2)

    R = torch.cat((r1, r2, r3), dim=0)
    return s, R, t3d

def matrix2angle(R):
    #assert (isRotationMatrix(R))

    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.stack((x, y, z))

def normalizeImg(img):
    min,max = torch.min(img),torch.max(img)
    img = (img -min)/(max-min)
    return img

# crop images with dlib
def rect2numpy(face,H,W):
    top = max(0, face.top())
    bottom = min(face.bottom(), H)
    left = max(0, face.left())
    right = min(face.right(), W)
    return [left,top,right,bottom]

# detector = dlib.get_frontal_face_detector()
# def face_parsing(imgs):
#     rects = []
#     H,W = imgs.size()[-2:]
#     images = normalizeImg(imgs.clone().detach().cpu()).numpy().transpose(0,2,3,1)
#     images = (images*255).astype('uint8')
#     for item in images:
#         gray = cv2.cvtColor(item, cv2.COLOR_RGB2GRAY)
#         rect = detector(gray, 1)
#         if 0 == len(rect):
#             rects.append([0,0,W,H])
#         else:
#             rects.append(rect2numpy(rect[0],H,W))
#     return rects

def crop_with_bbox(imgs,bboxs,target_size):
    images = []
    for img,bbox in zip(imgs,bboxs):
        images.append(torch.nn.functional.interpolate(img[None,:,bbox[1]:bbox[3],bbox[0]:bbox[2]], size=target_size,mode='bilinear', align_corners=True))
    return torch.cat(images,dim=0)

def attribute_loss(params_predict,params_real,param_whitening,condition_dim):
    loss1,loss2,loss3 = 0,0,0
    if params_real.dim() != params_predict.dim():
        params_real = params_real.squeeze(0)
    poses,shape,expression = parse_params(params_predict,param_whitening)
    loss1 = F.l1_loss(poses,params_real[...,:3]).mean()
    if condition_dim > 3:
        loss2 =  F.l1_loss(shape,params_real[...,3:43]).mean()
    else:
        loss2 = loss1
    if condition_dim > 34:
        loss3 = F.l1_loss(expression, params_real[...,43:]).mean()
    else:
        loss3 = loss1
    return loss1,loss2,loss3

def attribute_loss2(params_predict,params_real):
    loss = F.l1_loss(params_predict, params_real).mean()
    return loss

def condition_compete_loss(params):
    batch_size = params.size(0)
    params_real = params.clone().detach()
    index = torch.tensor([range(batch_size//2,batch_size),range(batch_size//2)]).flatten()
    loss = F.l1_loss(params[:,52:], params_real[index,52:]).mean()

    return loss

def identity_loss(embeddings):
    batch_size = embeddings.size(0)
    embeddings_real = embeddings.clone().detach()
    index = torch.tensor([range(batch_size // 2, batch_size), range(batch_size // 2)]).flatten()
    loss = F.l1_loss(embeddings, embeddings_real[index]).mean()

    return loss

def l1Loss(sec,target,mask):
    dim = target.size(1)
    return F.l1_loss(sec.view(target.size())[mask.repeat(1,dim,1,1)],target[mask.repeat(1,dim,1,1)]).mean()

class OhemCELoss(nn.Module):
    def __init__(self, n_min, thresh=0.7,  ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):

        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


##################################  GAN  Loss  ###########################################

def GAN_feat_loss(pred_fake,pred_real,lambda_feat=1.0):
    loss_G_GAN_Feat = 0
    feat_weights = 4.0 / len(pred_fake)
    for i in range(len(pred_fake)):
        loss_G_GAN_Feat += feat_weights * l1Loss(pred_fake[i], pred_real[i].detach()) * lambda_feat
    return loss_G_GAN_Feat

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

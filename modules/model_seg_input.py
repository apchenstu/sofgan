import math
from torch import nn
from utils import *
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class PixelNorm(nn.Module):
    def __init__(self, with_condition=False):
        super().__init__()
        self.with_condition = with_condition

    def forward(self, input):
        if self.with_condition:
            dim = input.size(1)
            normalize_input_1 = input[:,:dim//2] * torch.rsqrt(torch.mean(input[:,:dim//2] ** 2, dim=1, keepdim=True) + 1e-8)
            normalize_input_2 = input[:, dim // 2:] * torch.rsqrt(torch.mean(input[:, dim // 2:] ** 2, dim=1, keepdim=True) + 1e-8)
            return torch.cat((normalize_input_1,normalize_input_2),dim=1)
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def pixelNorm(input):
    return input * torch.rsqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + 1e-8)


def mask_features(feat,style_mask,batch):

    h, w = feat.shape[2:]
    if style_mask.shape[2] != h:
        style_mask = F.interpolate(style_mask, (h, w), mode='bilinear',align_corners=True)
    style_mask = style_mask[:, :, None, :, :]

    feat = torch.sum(feat.view(batch, 2, -1, h, w) * style_mask, 1)
    return feat

def index_one_noise_with_seg(segmap, noise1):
    batch = segmap.shape[0]
    _, c, height, width = noise1.size()

    if height != noise1.shape[0] or width != noise1.shape[1]:
        segmap = F.interpolate(segmap, size=(height, width), mode='nearest').long()

    noise1 = torch.gather(noise1.repeat(batch,1,1,1),1,segmap)

    return noise1

def index_noise_with_seg(segmap, noise1, noise2):
    batch = segmap.shape[0]
    _, c, height, width = noise1.size()

    if height != noise1.shape[0] or width != noise1.shape[1]:
        segmap = F.interpolate(segmap, size=(height, width), mode='nearest').long()

    noise1 = torch.gather(noise1.repeat(batch,1,1,1),1,segmap)
    noise2 = torch.gather(noise2.repeat(batch,1,1,1), 1, segmap)

    return noise1, noise2

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

def tensor_shift(seg, shift_x, shift_y):
    h, w = seg.size()[2:]

    # y direction
    if shift_y > 0:
        seg[:, :, shift_y:h] = seg[:, :, :(h - shift_y)].clone()
        seg[:, :, :shift_y] = 0
    else:
        seg[:, :, :(h + shift_y)] = seg[:, :, abs(shift_y):h].clone()
        seg[:, :, (h + shift_y):] = 0

    # x direction
    if shift_x > 0:
        seg[..., shift_x:w] = seg[..., :(w - shift_x)].clone()
        seg[..., :shift_x] = 0
    else:
        seg[..., :(w + shift_x)] = seg[..., abs(shift_x):w].clone()
        seg[..., (w + shift_x):] = 0
    return seg

def build_position(height, width):
    grad = torch.ones(height, width).nonzero().view(height, width, 2).unsqueeze(0).float().cuda()[..., [1, 0]]
    grad[..., 0], grad[..., 1] = grad[..., 0] / (width-1), grad[..., 1] / (height-1)
    return grad

def feature_remap(tensor, remap):
    # remap: [N,H,W,2]
    N,C,H,W = tensor.shape
    remap = F.interpolate(remap.permute(0,3,1,2), size=(H,W), mode='bilinear',align_corners=True).permute(0,2,3,1)
    grid = (build_position(H,W).repeat(N,1,1,1)+ remap*(H/128.0))*2-1.0
    return F.grid_sample(tensor,grid,align_corners=True)

def normalize_with_seg(out,style_feat):
    B,C,H,W = style_feat.shape
    gamma,beta = style_feat[:,:C//2],style_feat[:,C//2:]
    if gamma.dim() < 4:
        gamma,beta = gamma.unsqueeze(1),beta.unsqueeze(1)

    out = out*(1+gamma) + beta
    return out


def label_wise_conv2d(x, labels, weight, dilation=1, padding=0, stride=1, groups=1):
    '''
    :param input: [N,in_channel,H,W]
    :param labels: [N,1,H,W]
    :param weight: [out_channel,in_channel,kernel_h,kernel_w]
    :return: [N,out_channel,H,W]
    '''

    B, C, H, W = x.shape
    out_channel, in_channel, kernel_h, kernel_w = weight.shape

    out_h, out_w = int((H - kernel_h + padding * 2) / stride + 1), int((W - kernel_w + padding * 2) / stride + 1)
    inp_unf = F.unfold(x, (kernel_h, kernel_w), dilation, padding, stride)  # (N,in_channel*kernel_h*kernel_w,L)

    if 1 == groups:
        inp_unf = inp_unf.transpose(1, 2)
        weight = weight.view(weight.size(0), -1).t()

    elif groups > 1 and B == 1:
        inp_unf = inp_unf.view(groups, in_channel * kernel_h * kernel_w, -1).transpose(1, 2)
        weight = weight.view(groups, -1, in_channel * kernel_h * kernel_w).transpose(1, 2)
    else:
        exit()

    out_channel = out_channel // groups
    out_unf = inp_unf.matmul(weight).transpose(1, 2)
    out = F.fold(out_unf, (out_h, out_w), (1, 1)).view(-1, out_channel, out_h, out_w)
    return out


def add_stride(x, stride):
    B, C, H, W = x.shape
    x = torch.cat((x, torch.zeros((B, C, 1, W), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)), 2)
    x = torch.cat((x, torch.zeros((B, C, H + 1, 1), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)), 3)

    H_shuffl_index = torch.arange(0, H).short().repeat(stride, 1)
    W_shuffl_index = torch.arange(0, W).short().repeat(stride, 1)
    H_shuffl_index[1:], W_shuffl_index[1:] = H, W
    H_shuffl_index = H_shuffl_index.t().reshape(-1).tolist()[:-stride + 1]
    W_shuffl_index = W_shuffl_index.t().reshape(-1).tolist()[:-stride + 1]

    x = x[:, :, H_shuffl_index][..., W_shuffl_index]
    return x

def label_wise_transpose_conv2d(x, labels, weight, dilation=1, padding=0, stride=1, groups=1):
    '''
      :param input: [N,in_channel,H,W]
      :param labels: [N,1,H,W]
      :param weight: [in_channel, out_channel,kernel_h,kernel_w]
      :return: [N,out_channel,H,W]
      '''

    B, C, H, W = x.shape
    weight = torch.flip(weight, [2, 3])
    out_channel, in_channel, kernel_h, kernel_w = weight.shape

    if stride > 1:
        x = add_stride(x, stride)

    pad_width, pad_height = kernel_w - padding - 1, kernel_h - padding - 1
    x_pad = F.pad(x, (pad_width, pad_width, pad_height, pad_height))
    return label_wise_conv2d(x_pad, labels, weight, groups=groups)

def normalize_with_siw(x, segmap, gamma=None):
    segmap = scatter(segmap, label_size=x.size()[2:]).bool()

    b_size = x.shape[0]
    f_size = x.shape[1]
    s_size = segmap.shape[1]

    for i in range(b_size):
        for j in range(s_size):
            component_mask_area = torch.sum(segmap[i, j])
            if component_mask_area > 0:
                mask_feature = x[i].masked_select(segmap[i, j]).view(f_size, component_mask_area)
                variance = torch.std(mask_feature, dim=1, keepdim=True)
                mask_feature /= (variance+1e-6)
                if gamma is not None:
                    mask_feature *= (1.0+gamma[j])
                    # mask_feature += gamma[1, j]
    return x

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if 'fused_lrelu' == self.activation:
            negative_slope = 0.2
            scale = 2 ** 0.5
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul,negative_slope,scale)
        elif 'tanh' == self.activation:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
            out = torch.tanh(out)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class SegEncoder(nn.Module):
    def __init__(self,in_channel, blur_kernel=[1, 7, 7, 1]):
        super().__init__()
        kernel_size = 3
        pw = kernel_size // 2
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(segClass, nhidden, kernel_size=kernel_size, padding=pw),
            nn.LeakyReLU(0.2),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, in_channel, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, in_channel, kernel_size=kernel_size, padding=pw)
        # self.blur = Blur(blur_kernel, pad=(2, 1))

    def forward(self, style_img, size=(64, 64), shift=None):
        # style_img = scatter(style_img)
        # style_img = F.interpolate(style_img, size=size, mode='bilinear', align_corners=True)

        actv = self.mlp_shared(style_img)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # gamma, beta = self.blur(gamma), self.blur(beta)
        if shift is not None:
            height, width = size
            gamma,beta = tensor_shift(gamma,int(shift[0]*width/512),int(shift[1]*height/512)), \
                         tensor_shift(beta,int(shift[0]*width/512),int(shift[1]*height/512))

        return torch.cat((gamma,beta),dim=1)


# GAMMA,BETA = {},{}
class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        classwiseStyle=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.classwiseStyle = classwiseStyle
        if classwiseStyle:
            self.modulation = nn.ModuleList()
            for layer_idx in range(len(semanticGroups)):
                self.modulation.append(EqualLinear(style_dim, in_channel, bias_init=1))
        else:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate



    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style, segmap=None, labels=None):
        batch_style = style.shape[0]
        batch, in_channel, height, width = input.shape

        if self.classwiseStyle and segmap is not None and labels is not None:
            segmap_resized = F.interpolate(segmap.clone(), size=(height, width), mode='bilinear', align_corners=True)
            segmap_resized = F.softmax(segmap_resized, dim=1)

            probability = torch.zeros((batch, in_channel, height, width), device=input.device)

            if batch_style//batch != 1:
                style = style.view(batch, 2, -1).mean(1)

            for idx_batch, label in enumerate(labels):
                groups_idx = torch.unique(classBin[label]).tolist()
                for i,item in enumerate(groups_idx):
                    channelwise_scale = self.modulation[item](style[[idx_batch]]).view(in_channel, 1, 1)
                    if self.demodulate:
                        demod = torch.rsqrt(channelwise_scale.pow(2).sum() + 1e-8)
                        channelwise_scale = channelwise_scale * demod.view(1, 1, 1)

                    probability[idx_batch] = probability[idx_batch] + channelwise_scale \
                                             * torch.sum(segmap_resized[idx_batch,classBin[item]],dim=1,keepdim=True)
            input = input * probability

            weight = self.scale * self.weight
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(1, self.out_channel, 1, 1, 1)
            weight = weight.view(
                self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )

            if self.upsample:
                weight = weight.permute(1, 0, 2, 3)
                out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=1)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)
                out = self.blur(out)

            elif self.downsample:

                input = self.blur(input)
                _, _, height, width = input.shape
                out = F.conv2d(input, weight, padding=0, stride=2, groups=1)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)

            else:
                out = F.conv2d(input, weight, padding=self.padding, groups=1)
                _, _, height, width = out.shape
                out = out.view(batch, self.out_channel, height, width)

        else:
            style = self.modulation(style).view(batch_style, 1, in_channel, 1, 1)
            weight = self.scale * self.weight * style

            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch_style, self.out_channel, 1, 1, 1)

            weight = weight.view(
                batch_style * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )


            if self.upsample:

                input = input.view(1, batch * in_channel, height, width)
                weight = weight.view(
                    batch, batch_style//batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
                )
                weight = weight.permute(0, 3, 1, 2, 4, 5).reshape(
                    batch * in_channel, batch_style//batch * self.out_channel, self.kernel_size, self.kernel_size
                )
                out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch_style, self.out_channel, height, width)
                out = self.blur(out)


            elif self.downsample:

                input = self.blur(input)
                _, _, height, width = input.shape
                input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight,padding=0, stride=2, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch_style, self.out_channel, height, width)

            else:
                input = input.view(1, batch * in_channel, height, width)
                out = F.conv2d(input, weight, padding=self.padding, groups=batch)
                _, _, height, width = out.shape
                out = out.view(batch_style, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if noise.shape[0] != image.shape[0] and with_classwise_noise:
            noise_shape = noise.shape
            noise = noise.repeat(1,2,1,1).view(2*noise_shape[0],noise_shape[1],noise_shape[2],noise_shape[3])

        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        with_condition_img=False,
        style_merge=False,
        with_noise = True,
        classwiseStyle=False,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            downsample=downsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            classwiseStyle=classwiseStyle,
        )
        self.style_merge = style_merge
        self.with_condition_img = with_condition_img
        if with_condition_img:
            self.segEncoder = SegEncoder(out_channel)
        self.with_noise = with_noise
        if with_noise:
            self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None,style_mask=None,segmap=None, labels=None):
        batch,style_batch = input.shape[0], style.shape[0]
        if self.style_merge:
            b, dim = style.shape
            style_merged = torch.sum(style.view(batch, -1, dim),dim=1)
            out = self.conv(input, style_merged, segmap, labels)
        else:
            out = self.conv(input, style, segmap, labels)


        if style_mask is not None and not self.style_merge and batch != out.shape[0]:
            if not style_batch == batch*2:
                raise AssertionError('two times batch size not equal style size')

            out = mask_features(out, style_mask, batch)

        if self.with_noise:
            out = self.noise(out, noise=noise)
        out = self.activate(out)

        if self.with_condition_img and segmap is not None:
            H,W = out.shape[2:]
            segmap_resized = F.interpolate(segmap.clone(), size=(H, W), mode='bilinear', align_corners=True)
            spade = self.segEncoder(segmap_resized)
            out = normalize_with_seg(out, spade)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], with_condition_img=False, classwiseStyle=False):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, final_channel, 1, style_dim, demodulate=False, classwiseStyle=classwiseStyle)
        self.bias = nn.Parameter(torch.zeros(1, final_channel, 1, 1))

        if with_condition_img:
            kernel_size = 3
            pw = kernel_size // 2
            nhidden = 128
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(segClass, nhidden, kernel_size=kernel_size, padding=pw),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv2d(nhidden, 1, kernel_size=kernel_size, padding=pw)
            self.mlp_beta = nn.Conv2d(nhidden, 1, kernel_size=kernel_size, padding=pw)

    def forward(self, input, style, skip=None, style_mask=None, segmap=None, labels=None):
        batch = input.shape[0]
        out = self.conv(input, style, segmap, labels)

        if style_mask is not None:
            if not style.shape[0] == batch*2:
                raise AssertionError('two times batch size not equal style size')
            out = mask_features(out, style_mask, batch)


        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )


    def forward(self, input, styles=None, style_mask=None):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        size = args.resolution
        channel_multiplier = args.channel_multiplier

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(final_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def load_my_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input):
        D_feat = []
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group =batch
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        D_feat.append(out)
        out = self.final_conv(out)
        D_feat.append(out)
        out = out.view(batch, -1)
        D_feat.append(out)
        out = self.final_linear(out)

        return out, D_feat

#####################################################################################################################
# 0305 update
# modify encoder into pixel noise, remove layered std and bias
#####################################################################################################################

class Generator(nn.Module):
    def __init__(
        self,
        args,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.args = args
        style_dim = args.latent
        n_mlp = args.n_mlp
        channel_multiplier = args.channel_multiplier
        self.size = args.resolution
        self.style_dim = style_dim
        self.spadeLayer = 64

        with_condition = self.args.condition_path is not None

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        init_channels = segClass

        self.segEncoder = nn.ModuleList()
        filters = [init_channels, 32, 64, 128]
        for layer in range(len(filters)-1):
            downsample = True
            self.segEncoder.append(ResBlock(filters[layer], filters[layer+1],downsample=downsample))


        layers = [PixelNorm(with_condition)] + [EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )]
        for i in range(n_mlp-1):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)


        self.const_size = 4
        self.conv1 = StyledConv(
            filters[-1], self.channels[self.const_size], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[self.const_size], style_dim, upsample=False)

        self.log_size = int(math.log(self.size, 2))
        self.num_layers = (self.log_size - int(math.log(self.const_size, 2))) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            res = int((1+layer_idx) // 2 + math.log(self.const_size, 2))
            shape = [1, 1, 2 ** res, 2 ** res] if with_classwise_noise else [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        in_channel = self.channels[self.const_size]
        for i in range(int(math.log(self.const_size,2)), self.log_size):
            resolution = 2 ** (i + 1)
            out_channel = self.channels[resolution]
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    with_condition_img=(resolution >= 64 and resolution<=256),
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel#, with_condition_img=(resolution >= 64 and resolution<=256)
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = int(self.log_size - math.log(self.const_size,2) +1)*2

    def make_noise(self):
        noises = []
        for layer_idx in range(self.num_layers):
            res = int((1+layer_idx) // 2 + math.log(self.const_size, 2))
            shape = [1, 1, 2 ** res, 2 ** res] if with_classwise_noise else [1, 1, 2 ** res, 2 ** res]
            noises.append(torch.randn(shape).cuda())

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def load_my_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def style_map_norepeat(self,styles,segmap=None):
        styles_w = []
        if styles.dim()<=2:
            styles = styles.unsqueeze(0)

        if segmap is not None:
            out = segmap
            out = F.interpolate(out, (32, 32), mode='bilinear', align_corners=True)
            for filter in self.segEncoder:
                out = filter(out)

        for s in styles:
            if s.dim()>2:
                s = s.view(-1,s.shape[-1])
            styles_w.append(self.style(s))
        return torch.cat(styles_w,dim=0)

    def style_map(self,styles,truncation=1,truncation_latent=None,inject_index=None,to_w_space=True):

        if to_w_space:
            styles_w = []
            for s in styles:
                if s.dim()>2:
                    s = s.view(-1,s.shape[-1])
                styles_w.append(self.style(s))
        else:
            styles_w = styles


        if truncation < 1:
            style_t = []

            for style in styles_w:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles_w = style_t


        if len(styles_w) < 2:
            inject_index = self.n_latent

            if styles_w[0].ndim < 3:
                latent = styles_w[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles_w[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles_w[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles_w[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        return latent

    def forward(
        self,
        styles,
        condition_img=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=False,
        return_intermedia=False,
        style_mask=None,
    ):
        featuresMaps,parsing_feature = [],[]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        labels = []
        batch = condition_img.shape[0] if condition_img is not None else styles.shape[0]
        if condition_img is not None:
            for temp in condition_img:
                labels.append(torch.unique(temp.long()))

        if not input_is_latent:
            latent = self.style_map(styles, truncation, truncation_latent, inject_index)
        else:
            latent = styles
            if latent.dim() > 3:
                latent = latent.reshape(-1, latent.shape[-2], latent.shape[-1])


        condition_img = scatter(condition_img, label_size=(128, 128))
        batch_style = styles.shape[-3]*styles.shape[-2] if not input_is_latent else styles.shape[0]
        if batch != batch_style and style_mask is None:
            style_mask = scatter_to_mask(condition_img.clone(), labels)



        out = condition_img
        out = F.interpolate(out, (32, 32), mode='bilinear', align_corners=True)
        for filter in self.segEncoder:
            out = filter(out)

        parsing_feature = out

        noise0 = index_one_noise_with_seg(condition_img, noise[0]) if with_classwise_noise else noise[0]
        out = self.conv1(out, latent[:, 0], noise=noise0, style_mask=style_mask)
        skip = self.to_rgb1(out, latent[:, 1], style_mask=style_mask)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1, style_mask=style_mask, segmap=condition_img, labels=labels)
            out = conv2(out, latent[:, i + 1], noise=noise2, style_mask=style_mask, segmap=condition_img, labels=labels)
            skip = to_rgb(out, latent[:, i + 2], skip, style_mask=style_mask, segmap=condition_img, labels=labels)

            i += 2
            if return_intermedia:
                featuresMaps.append(skip.cpu().detach())

        image = skip

        if return_latents:
            return image, latent, featuresMaps, parsing_feature

        else:
            return image, None, featuresMaps, parsing_feature


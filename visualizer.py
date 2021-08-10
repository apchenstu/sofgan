import torch
import numpy as np
from torchvision import transforms
from PIL import ImageDraw

def draw(img, corner, imgpts):
    draw = ImageDraw.Draw(img)
    imgpts[:, 1] = -imgpts[:, 1]
    imgpts += np.array(corner)
    draw.line(corner + tuple(imgpts[0].ravel()), fill= (255, 0, 0), width=3)
    draw.line(corner+ tuple(imgpts[1].ravel()), fill=(0, 255, 0), width=3)
    draw.line(corner+ tuple(imgpts[2].ravel()), fill=(0, 0, 255), width=3)
    # img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 10)
    # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 10)
    # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 10)
    return img

def angle2matrix(pose):
    ''' compute Rotation Matrix from three Euler angles
    '''
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pose[0]), -np.sin(pose[0])],
                    [0, np.sin(pose[0]), np.cos(pose[0])]
                    ])

    R_y = np.array([[np.cos(pose[1]), 0, np.sin(pose[1])],
                    [0, 1, 0],
                    [-np.sin(pose[1]), 0, np.cos(pose[1])]
                    ])

    R_z = np.array([[np.cos(pose[2]), -np.sin(pose[2]), 0],
                    [np.sin(pose[2]), np.cos(pose[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

import matplotlib.pyplot as plt
def normalize(features):
    # tensor [B,C,H,W]
    B, C, H, W = features.shape
    result = []
    cm = plt.get_cmap('jet')
    for i in range(B):
        tensor = features[i]
        tensor = torch.abs(tensor).sum(dim=0)
        mu,sigma = torch.mean(tensor,0,keepdim=True),torch.var(tensor,0,keepdim=True)
        tensor = (tensor - mu) / sigma
        tensor = torch.clamp(tensor, min=0, max=1.0)
        colored_image = torch.tensor(cm(tensor)[...,:3]).permute(2,0,1)

        result.append(colored_image*2-1.0)
    return torch.stack(result,dim=0)

def normalizeImg(img):
    min,max = torch.min(img),torch.max(img)
    img = (img -min)/(max-min)
    return img

def draw_pose(samples,poses):
    # samples N*H*W*3
    # poses   N*3 Eurler
    imgs, samples = [], samples
    for img,pose in zip(samples,poses[0].unbind(0)):
        H,W = img.shape[1:]
        rotate = angle2matrix(pose.cpu().numpy())
        axis = H / 4 * np.float32([[1, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(-1, 3)
        imgpts = np.dot(rotate, axis.T).T
        imgpts = imgpts[:, :2].astype('int')
        center = (W // 2, H // 2)
        img = draw(transforms.ToPILImage()(normalizeImg(img)), center, imgpts)
        imgs.append(transforms.ToTensor()(img)*2-1)
    imgs = torch.from_numpy(np.stack(imgs,axis=0))
    return imgs

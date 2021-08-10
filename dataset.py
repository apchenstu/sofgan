from io import BytesIO

import lmdb, os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import torch
import numpy as np

def getListOfFiles(dirName):
    filelist = []

    for root, dirs, files in os.walk(dirName):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))

    return filelist

remap_raw_to_new_list = torch.tensor([0,1,2,3,4,5,14,11,12,13,6,7,8,9,10,15,16,17,18,19]).float()
def id_raw_to_new(seg):
    #raw ['background'0, 'skin'1, 'l_brow'2, 'r_brow'3, 'l_eye'4, 'r_eye'5, 'eye_g'6, 'l_ear'7, 'r_ear'8, 'ear_r'9,
    #'r_nose'10 'l_nose'11,'mouth'12, 'u_lip'13, 'l_lip'14, 'neck'15, 'neck_l'16, 'cloth'17, 'hair'18, 'hat'19]
    return remap_raw_to_new_list[seg.long()]

remap_list = torch.tensor([0,1,2,2,3,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16]).float()
def id_remap(seg):
    #['background'0,'skin'1, 'l_brow'2, 'r_brow'3, 'l_eye'4, 'r_eye'5,'r_nose'6, 'l_nose'7, 'mouth'8, 'u_lip'9,
    # 'l_lip'10, 'l_ear'11, 'r_ear'12, 'ear_r'13, 'eye_g'14, 'neck'15, 'neck_l'16, 'cloth'17, 'hair'18, 'hat'19]

    return remap_list[seg.long()]

remap_list2 = torch.tensor([0, 1,2,3,5,4,6,7,8,9,10,11,12,13,14,15,16]).float()
def flip_labels(seg):
        return remap_list2[seg.long()]

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, condition_path=None):
        if 'LMDB' in path:
            self.env = lmdb.open(
                path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            if not self.env:
                raise IOError('Cannot open lmdb dataset', path)
            with self.env.begin(write=False) as txn:
                self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
                if self.length > 70000:
                    self.length = 70000
        else:

            folders = path.split(',')
            self.imgs_list = []
            for folder in folders:
                self.imgs_list += getListOfFiles(folder)

            self.length = len(self.imgs_list)


        self.has_condition = False
        if condition_path is not None:
            if 'LMDB' in condition_path:
                self.condition = lmdb.open(
                    condition_path,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                if not self.condition:
                    raise IOError('Cannot open lmdb condition dataset', condition_path)

                with self.condition.begin(write=False) as txn:
                    self.length_seg = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            else:

                folders = condition_path.split(',')
                self.segmap_list = []
                for folder in folders:
                    self.segmap_list += getListOfFiles(folder)

                self.length_seg = len(self.segmap_list)

            self.has_condition = True
            self.condition_transform = transforms.Compose([transforms.ToTensor()])#

        self.resolution = resolution
        self.transform = transform
        self.img_root, self.seg_root = path, condition_path


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        resizeScale = np.random.rand() * 0.5

        if 'LMDB' in self.img_root:
            # key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            key = f'{1024}-{str(index).zfill(5)}'.encode('utf-8')
            with self.env.begin(write=False) as txn:
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)

        else:
            img = Image.open(os.path.join(self.img_root, self.imgs_list[index])).convert('RGB')
        img = img.resize((int(self.resolution * (1.0 + resizeScale)), int(self.resolution * (resizeScale + 1.0))))
        img = self.transform(img)

        codition_img = []
        if self.has_condition:
            index_seg = int(index * self.length_seg / self.length)
            if 'LMDB' in self.seg_root:
                key = f'{str(index_seg).zfill(5)}'.encode('utf-8')
                with self.condition.begin(write=False) as txn:
                    condition_bytes = txn.get(key)
                    buffer = BytesIO(condition_bytes)
                    codition_img = Image.open(buffer)
            else:
                codition_img = Image.open(os.path.join(self.seg_root, self.segmap_list[index_seg]))
            codition_img = codition_img.resize((int(self.resolution * (1.0 + resizeScale)), int(self.resolution * (1.0 + resizeScale))),
                                                        resample =Image.NEAREST)
            codition_img = self.condition_transform(codition_img) * 255  # id_raw_to_new
            if 'LMDB' in self.seg_root:
                codition_img = id_raw_to_new(codition_img)
            codition_img = id_remap(codition_img)


        if torch.rand(1) > 0.5:
            img = img.flip(2)
            if self.has_condition:
                codition_img = flip_labels(codition_img).flip(2)

        top_left = [int(np.random.rand() * resizeScale * self.resolution), int(np.random.rand() * resizeScale * self.resolution)]
        img = img[:,top_left[0]:top_left[0]+self.resolution,top_left[1]:self.resolution+top_left[1]]
        if self.has_condition:
            codition_img = codition_img[:,top_left[0]:top_left[0]+self.resolution,top_left[1]:self.resolution+top_left[1]]

        return img, codition_img

class DatasetSimple(Dataset):
    def __init__(self, path, transform, resolution=256, condition_path=None):
        if 'LMDB' in path:
            self.env = lmdb.open(
                path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            if not self.env:
                raise IOError('Cannot open lmdb dataset', path)
            with self.env.begin(write=False) as txn:
                self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
                if self.length > 70000:
                    self.length = 70000
        else:

            folders = path.split(',')
            self.imgs_list = []
            for folder in folders:
                self.imgs_list += getListOfFiles(folder)

            self.length = len(self.imgs_list)


        self.has_condition = False
        if condition_path is not None:
            if 'LMDB' in condition_path:
                self.condition = lmdb.open(
                    condition_path,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                if not self.condition:
                    raise IOError('Cannot open lmdb condition dataset', condition_path)

                with self.condition.begin(write=False) as txn:
                    self.length_seg = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            else:

                folders = condition_path.split(',')
                self.segmap_list = []
                for folder in folders:
                    self.segmap_list += getListOfFiles(folder)

                self.length_seg = len(self.segmap_list)

            self.has_condition = True
            self.condition_transform = transforms.Compose([ transforms.Resize(256,interpolation=0),transforms.ToTensor()])#

        self.resolution = resolution
        self.transform = transform
        self.img_root, self.seg_root = path, condition_path


    def __len__(self):
        return self.length

    def __getitem__(self, index):

        if 'LMDB' in self.img_root:
            # key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            key = f'{1024}-{str(index).zfill(5)}'.encode('utf-8')
            with self.env.begin(write=False) as txn:
                img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)

        else:
            img = Image.open(os.path.join(self.img_root, self.imgs_list[index])).convert('RGB')

        img = self.transform(img)

        codition_img = []
        if self.has_condition:
            index_seg = int(index * self.length_seg / self.length)
            if 'LMDB' in self.seg_root:
                key = f'{str(index_seg).zfill(5)}'.encode('utf-8')
                with self.condition.begin(write=False) as txn:
                    condition_bytes = txn.get(key)
                    buffer = BytesIO(condition_bytes)
                    codition_img = Image.open(buffer)
            else:
                codition_img = Image.open(os.path.join(self.seg_root, self.segmap_list[index_seg]))

            codition_img = self.condition_transform(codition_img) * 255  # id_raw_to_new
            if 'LMDB' in self.seg_root:
                codition_img = id_raw_to_new(codition_img)
            codition_img = id_remap(codition_img)


        if torch.rand(1) > 0.5:
            img = img.flip(2)
            if self.has_condition:
                codition_img = flip_labels(codition_img).flip(2)


        return img, codition_img

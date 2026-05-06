"""
CD Dataset
"""
import os
from PIL import Image
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import torchvision
import torch

totensor = torchvision.transforms.ToTensor()

def transform_augment_cd(img, min_max=(0, 1)):
    img = totensor(img)
    return img * (min_max[1] - min_max[0]) + min_max[0]

def genreate_list(root, split):
    list_path = os.path.join(root, split+'.txt')
    with open(list_path, 'w') as f:
        for img_name in os.listdir(os.path.join(root)):
            f.write(img_name + '\n')

second_colormap = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]

"""
CD Dataset 
├─image
├─image_post
├─label
└─list
"""

IMG_FOLDER_NAME = 'A'
IMG_POST_FOLDER_NAME = 'B'
LABEL_FOLDER_NAME = 'label'
LABEL1_FOLDER_NAME = 'label1'
LABEL2_FOLDER_NAME = 'label2'
LIST_FOLDER_NAME = 'list'

label_suffix = ".png"

#list内存放image_name 构建读取图片名字函数
def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

#获取各个文件夹的路径
def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL_FOLDER_NAME, img_name)

def get_label1_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL1_FOLDER_NAME, img_name)

def get_label2_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL2_FOLDER_NAME, img_name)

class CDDataset(Dataset):
    def __init__(self, root_dir, resolution=256, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split #train / val / test
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_label = Image.open(L_path).convert("RGB")

        img_A = transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = transform_augment_cd(img_B, min_max=(-1, 1))
        img_label = transform_augment_cd(img_label, min_max=(0, 1))
        if img_label.dim() > 2:
            img_label = img_label[0]

        return {'A':img_A, 'B':img_B, 'L':img_label, 'Index':index}


class SCDDataset(Dataset):
    def __init__(self, root_dir, resolution=512, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])
        name = A_path.split('\\')[-1].split('.')[0]
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        L1_path = get_label1_path(self.root_dir, self.img_name_list[index % self.data_len])
        L2_path = get_label2_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_label = np.array(Image.open(L_path), dtype=np.uint8)
        img_label1 = np.array(Image.open(L1_path), dtype=np.uint8)
        img_label2 = np.array(Image.open(L2_path), dtype=np.uint8)

        img_A = transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = transform_augment_cd(img_B, min_max=(-1, 1))
        img_label = torch.from_numpy(img_label)
        img_label1 = torch.from_numpy(img_label1)
        # add cls label on label1
        cls_category1 = torch.unique(img_label1)
        cls_label1 = torch.zeros(7, dtype = int)
        for index in cls_category1:
            cls_label1[int(index)] = 1

        img_label2 = torch.from_numpy(img_label2)
        # add cls label on label2
        cls_category2 = torch.unique(img_label2)
        cls_label2 = torch.zeros(7, dtype=int)
        for index in cls_category2:
            cls_label2[int(index)] = 1

        if img_label.dim() > 2:
            img_label = img_label[0]
            img_label1 = img_label1[0]
            img_label2 = img_label2[0]

        return {'A':img_A, 'B':img_B, 'L':img_label, 'L1':img_label1, 'L2':img_label2,
                'Index':index, 'name':name, 'cls1':cls_label1, 'cls2':cls_label2}

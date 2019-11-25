from __future__ import print_function, division
import os, cv2, torch
from PIL import Image
import pandas as pd
import numpy as np
import utils.custom_transforms as tr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.datasetPath import Path

# 전처리
def transform_tr(self, sample):
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
        tr.FixScaleCrop(crop_size=self.args.crop_size),
        tr.RandomGaussianBlur(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)

def transform_val(self, sample):
    composed_transforms = transforms.Compose([
        tr.FixScaleCrop(crop_size=self.args.crop_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)

def train_dataloader(storageType = 'local', input_size=224, batch_size=64, num_workers=0,):
    datasetPath = Path.db_root_dir(storageType)
    if storageType == 'local':
        image_dir = os.path.join(datasetPath , 'train', 'train_data', 'images')
        train_label_path = os.path.join(datasetPath , 'train', 'train_label')
        train_meta_path = os.path.join(datasetPath , 'train', 'train_data', 'train_with_valid_tags.csv')
        train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)

        dataloader = DataLoader(localData(
            image_dir, train_meta_data, label_path=train_label_path,
            transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
        return dataloader

    elif storageType == 'KHD_NSML':
        # 아래 경로는 데이터셋 형태에 따라 수정할것.
        image_dir = os.path.join(datasetPath, 'train', 'train_data', 'images')
        train_label_path = os.path.join(datasetPath, 'train', 'train_label')
        train_meta_path = os.path.join(datasetPath, 'train', 'train_data', 'train_with_valid_tags.csv')
        train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)

        dataloader = DataLoader(KHDdataset_NSML(
                            image_dir, train_meta_data, label_path=train_label_path,
                            transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
        return dataloader
    else:
        raise NotImplementedError

# 전처리를 적용한 데이터셋 반환
class KHDdataset_NSML(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform

        if self.label_path is not None:
            self.label_matrix = np.load(label_path)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(self.meta_data['package_id'].iloc[idx]),
                                str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load()  # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)

        if self.label_path is not None:
            tags = torch.tensor(
                np.argmax(self.label_matrix[idx]))  # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img

class localData(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform

        if self.label_path is not None:
            self.label_matrix = np.load(label_path)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(self.meta_data['package_id'].iloc[idx]),
                                str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load()  # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)

        if self.label_path is not None:
            tags = torch.tensor(
                np.argmax(self.label_matrix[idx]))  # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img
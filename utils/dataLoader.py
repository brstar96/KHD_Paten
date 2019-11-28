from __future__ import print_function, division
import os, cv2, torch, time
from PIL import Image
import pandas as pd
import numpy as np
import utils.custom_transforms as tr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 전처리
def crop_boxing_img(img_name, TRAIN_IMG_PATH, TEST_IMG_PATH, df_train, df_test, margin=-4, size=(224, 224)):
    # Bbox croping function for KaKR 3rd car classification competition
    if img_name.split('_')[0] == 'train':
        PATH = TRAIN_IMG_PATH
        data = df_train
    else:
        PATH = TEST_IMG_PATH
        data = df_test

    img = Image.open(os.path.join(PATH, img_name))
    pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    width, height = img.size
    x1 = max(0, pos[0] - margin)
    y1 = max(0, pos[1] - margin)
    x2 = min(pos[2] + margin, width)
    y2 = min(pos[3] + margin, height)

    return img.crop((x1, y1, x2, y2)).resize(size)

# 아래의 데이터셋 로더 클래스(KaKR3rdDataset, MammoDataset)에 정의(transform_tr/val)해두었지만 혹시몰라 남겨둠.
# def Mammo_preprocessing(data):
#     print('Preprocessing start')
#     # 자유롭게 작성해주시면 됩니다.
#     data = np.concatenate([np.concatenate([data[:, 0], data[:, 1]], axis=2)
#                               , np.concatenate([data[:, 2], data[:, 3]], axis=2)], axis=1)
#
#     X = np.expand_dims(data, axis=1)
#     X = X - X.min() / (X.max() - X.min())
#
#     print('Preprocessing complete...')
#     print('The shape of X changed', X.shape)
#
#     return X

# KaKR 3rd car classification competiton을 위한 custom dataloader
class KaKR3rdDataset(Dataset):
    def read_dataset(self):
        t = time.time()
        print('Data loading...')
        data_path = []  # data path 저장을 위한 변수
        labels = []  # 테스트 id 순서 기록
        ## 하위 데이터 path 읽기
        for dir_name, _, _ in os.walk(self.data_set_path):
            try:
                data_id = dir_name.split('/')[-1]
                int(data_id)
            except:
                pass
            else:
                data_path.append(dir_name)
                labels.append(int(data_id[0]))

        ## 데이터만 읽기
        data = []  # img저장을 위한 list
        for d_path in data_path:
            sample = np.load(d_path + '/mammo.npz')['arr_0']
            data.append(sample)
        data = np.array(data)  ## list to numpy

        print('Dataset Reading Success \n Reading time', time.time() - t, 'sec')
        print('Dataset:', data.shape, 'np.array.shape(files, views, width, height)')

        return data, labels, len(data), len(labels)  # Numpy arr과 정답 클래스

    def __init__(self, args, mode, DATA_PATH, transforms = None):
        self.args = args
        self.mode = mode
        self.data_set_path = DATA_PATH
        self.data, self.labels, self.len_data, self.len_label = self.read_dataset()

        self.len = self.data.shape[0]
        self.x_data = torch.from_numpy(self.data)
        self.y_data = torch.from_numpy(self.labels)

        # 전처리를 위한 transforms 초기화
        self.transforms = transforms

    def __getitem__(self, index):
        # Grayscale 등으로 변환할 경우 이곳에서 작업할것.

        if self.mode == 'train':
            return self.transform_tr(self.x_data)
        elif self.mode == 'val':
            return self.transform_val(self.x_data)
        else:
            print("Invalid params input")
            raise NotImplementedError

    def __len__(self):
        return self.len

    # custom_transforms.py의 전처리 항목을 적용한 transforms.Compose 클래스 반환
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
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


# KHD Mammo dataset을 위한 데이터로더
class MammoDataset(Dataset):
    def __init__(self, args, mode, DATA_PATH):
        self.args = args
        self.mode = mode
        self.data_set_path = DATA_PATH
        self.data, self.labels, self.len_data, self.len_label = self.read_dataset()

        self.len = self.data.shape[0]
        self.x_data = torch.from_numpy(self.data)
        self.y_data = torch.from_numpy(self.labels)

        # 전처리를 위한 transforms 초기화
        self.transforms = transforms

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.mode == 'train':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        else:
            print("Invalid params input")
            raise NotImplementedError

    def __len__(self):
        return self.len

    def read_dataset(self):
        t = time.time()
        print('Data loading...')
        data_path = []  # data path 저장을 위한 변수
        labels = []  # 테스트 id 순서 기록
        ## 하위 데이터 path 읽기
        for dir_name, _, _ in os.walk(self.data_set_path):
            try:
                data_id = dir_name.split('/')[-1]
                int(data_id)
            except:
                pass
            else:
                data_path.append(dir_name)
                labels.append(int(data_id[0]))

        ## 데이터만 읽기
        data = []  # img저장을 위한 list
        for d_path in data_path:
            sample = np.load(d_path + '/mammo.npz')['arr_0']
            data.append(sample)
        data = np.array(data)  ## list to numpy

        print('Dataset Reading Success \n Reading time', time.time() - t, 'sec')
        print('Dataset:', data.shape, 'np.array.shape(files, views, width, height)')

        return data, labels, len(data), len(labels)  # Numpy arr과 정답 클래스

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.x_data[index]).convert('RGB')
        _target = self.y_data[index]
        return _img, _target

    # custom_transforms.py의 전처리 항목을 적용한 transforms.Compose 클래스 반환
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
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
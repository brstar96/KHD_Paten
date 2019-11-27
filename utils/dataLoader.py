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

def train_dataloader(storageType = 'local', DATA_PATH = None, input_size=224, batch_size=64, num_workers=4,):
    if storageType == 'local':
        df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
        df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
        df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))

        # Bbox를 사용해 자른 이미지를 저장할 디렉토리
        TRAIN_CROPPED_PATH = '../cropped_train'
        TEST_CROPPED_PATH = '../cropped_test'

        # 잘린 이미지를 저장(저장할 경로가 없을 시 새 디렉토리 생성)
        if (os.path.isdir(TRAIN_CROPPED_PATH) == False):
            os.mkdir(TRAIN_CROPPED_PATH)
        if (os.path.isdir(TEST_CROPPED_PATH) == False):
            os.mkdir(TEST_CROPPED_PATH)
        for i, row in df_train.iterrows():
            cropped = crop_boxing_img(row['img_file'])
            cropped.save(os.path.join(TRAIN_CROPPED_PATH, row['img_file']))
        for i, row in df_test.iterrows():
            cropped = crop_boxing_img(row['img_file'])
            cropped.save(os.path.join(TEST_CROPPED_PATH, row['img_file']))

        df_train['class'] = df_train['class'].astype('str')
        df_train = df_train[['img_file', 'class']]
        df_test = df_test[['img_file']]

        dataloader = DataLoader(localData(
            image_dir, train_meta_data, label_path=train_label_path,
            transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
        return dataloader

    elif storageType == 'KHD_NSML':
        t = time.time()
        print('Data loading...')
        data_path = []  # data path 저장을 위한 변수
        labels = []  # 테스트 id 순서 기록
        ## 하위 데이터 path 읽기
        for dir_name, _, _ in os.walk(DATA_PATH):
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

        return data, labels
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
from __future__ import print_function, division
import os, cv2, torch, time
from PIL import Image
import numpy as np
import utils.custom_transforms as tr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from constants import VIEWS

# KaKR 3rd 데이터셋 전처리
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

# KaKR 3rd car classification competiton을 위한 custom dataloader
class KaKR3rdDataset(Dataset):
    def __init__(self, args, df, mode, DATA_PATH, transforms = None):
        self.args = args
        self.df = df
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
    def __init__(self, args, mode, data, labels, len_data, len_label):
        self.args = args
        self.mode = mode
        self.data, self.labels, self.len_data, self.len_label = data, labels, len_data, len_label
        self.LMLO, self.RMLO, self.LCC, self.RCC = self.preprocessing(self.data)
        # self.LMLO_x_data = torch.from_numpy(self.LMLO)
        # self.RMLO_x_data = torch.from_numpy(self.RMLO)
        # self.LCC_x_data = torch.from_numpy(self.LCC)
        # self.RCC_x_data = torch.from_numpy(self.RCC)
        # self.y_data = torch.from_numpy(self.labels)
        # self.x_data_views = [self.LMLO_x_data, self.RMLO_x_data, self.LCC_x_data, self.RCC_x_data]
        self.x_data_views = [self.LMLO, self.RMLO, self.LCC, self.RCC]
        self.y_data = self.labels

        # 전처리를 위한 transforms 초기화
        self.transforms = transforms

    def __getitem__(self, index):
        if self.mode == 'train':
            return {'image': self.transform_tr(self.x_data_views), 'label': self.y_data}
        elif self.mode == 'val':
            return {'image' : self.transform_val(self.x_data_views), 'label' : self.y_data}
        else:
            print("Invalid params input")
            raise NotImplementedError

    def __len__(self):
        return self.len_data

    # custom_transforms.py의 전처리 항목을 적용한 transforms.Compose 클래스 반환
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def clahe(self, img_arr):
        print("shape of img_arr : ", img_arr.shape) # (1800, 410, 333)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        PILim = Image.fromarray(img_arr) # [0,:]
        res = clahe.apply(np.array(PILim))
        return res  # Grayscale

    def postclahe(self, img):
        median_clahe = cv2.medianBlur(img, 5)
        ret, th1 = cv2.threshold(median_clahe, 127, 255, cv2.THRESH_BINARY)
        return th1  # Grayscale

    def flip_image(self, image, view, horizontal_flip):
        """
        If training mode, makes all images face right direction.
        In medical, keeps the original directions unless horizontal_flip is set.
        """
        if horizontal_flip == 'NO':
            if VIEWS.is_right(view):
                image = np.fliplr(image)
        elif horizontal_flip == 'YES':
            if VIEWS.is_left(view):
                image = np.fliplr(image)

        return image

    def preprocessing(self, data):
        print('Preprocessing start')
        # 11/30일 이슈 : 아래 LMLO의 shape를 찍어보면 (410, 333)이 나오나, clahe()로 넘긴 직후 shape를 찍어보면 (1800, 410, 333)이 나오는 버그(미해결)
        LMLO = data[:,0] # (1800, 1, 410, 333) - 범위로 인덱싱하지 않아서 생긴 문제인지 확인할것.
        print('shape of LMLO : ', LMLO.shape)
        LMLO_clahe = self.clahe(LMLO) # (410, 333)
        print('shape of LMLO_clahe : ', LMLO_clahe.shape)
        LMLO_postclahe = self.postclahe(LMLO_clahe) # (410, 333)
        print('shape of LMLO_postclahe : ', LMLO_postclahe.shape)
        final_LMLO = np.array(cv2.merge([LMLO, LMLO_clahe, LMLO_postclahe])).astype(np.float32).transpose((2, 0, 1)) # (3, 410, 333)

        RMLO = data[:,1] # (1800, 1, 410, 333)
        RMLO_clahe = self.clahe(RMLO)
        RMLO_postclahe = self.postclahe(RMLO_clahe)
        final_RMLO = np.array(cv2.merge([RMLO, RMLO_clahe, RMLO_postclahe])).astype(np.float32).transpose((2, 0, 1))  # (3, 410, 333)

        LCC = data[:,2] # (1800, 1, 410, 333)
        LCC_clahe = self.clahe(LCC)
        LCC_postclahe = self.postclahe(LCC_clahe)
        final_LCC = np.array(cv2.merge([LCC, LCC_clahe, LCC_postclahe])).astype(np.float32).transpose((2, 0, 1))  # (3, 410, 333)

        RCC = data[:,3] # (1800, 1, 410, 333)
        RCC_clahe = self.clahe(RCC)
        RCC_postclahe = self.postclahe(RCC_clahe)
        final_RCC = np.array(cv2.merge([RCC, RCC_clahe, RCC_postclahe])).astype(np.float32).transpose((2, 0, 1))  # (3, 410, 333)

        print('Preprocessing complete...')
        return final_LMLO, final_RMLO, final_LCC, final_RCC
# Code Written by Myeong-Gyu.Lee, Original code from https://www.kaggle.com/tmheo74/3rd-ml-month-car-image-cropping
from PIL import Image
import os
import pandas as pd
from ImgPreprocessing.UnzipDataset import unzip
from ImgPreprocessing.Utils_code import printProgress

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../../")
parent_list = (os.listdir(parent_dir))
DatasetPath = os.path.join(parent_dir, '2019-3rd-ml-month-with-kakr')

# 이미지 폴더 경로
TRAIN_IMG_PATH = os.path.join(DatasetPath, 'train')
TEST_IMG_PATH = os.path.join(DatasetPath, 'test')

# CSV 파일 경로
df_train = pd.read_csv(os.path.join(DatasetPath, 'train.csv'))
df_test = pd.read_csv(os.path.join(DatasetPath, 'test.csv'))
df_class = pd.read_csv(os.path.join(DatasetPath, 'class.csv'))

# Image Crop Function
def crop_boxing_img(img_name, TRAIN_IMG_PATH, TEST_IMG_PATH, df_train, df_test, margin=16):
    if img_name.split('_')[0] == "train":
        PATH = TRAIN_IMG_PATH
        data = df_train
    elif img_name.split('_')[0] == "test":
        PATH = TEST_IMG_PATH
        data = df_test

    img = Image.open(os.path.join(PATH, img_name))
    pos = data.loc[data["img_file"] == img_name, \
                   ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    width, height = img.size
    x1 = max(0, pos[0] - margin)
    y1 = max(0, pos[1] - margin)
    x2 = min(pos[2] + margin, width)
    y2 = min(pos[3] + margin, height)

    return img.crop((x1, y1, x2, y2))

def ImgCrop(parent_list, TRAIN_IMG_PATH, TEST_IMG_PATH, df_train, df_test):
    unzip(parent_list)
    Dataset_CroppedPath = [os.path.join(DatasetPath, 'train_cropped'), os.path.join(DatasetPath, 'test_cropped')]

    for i in range(len(Dataset_CroppedPath)):
        if not (os.path.isdir(Dataset_CroppedPath[i])):  # 디렉토리 존재여부 확인
            os.makedirs(Dataset_CroppedPath[i])
            print("New directory '" + str(Dataset_CroppedPath[i]) + "' has been created.")

    total_df_train = len(df_train)
    total_df_test = len(df_test)

    # Process Train Image Crop
    for i, row in df_train.iterrows():
        cropped = crop_boxing_img(row['img_file'], TRAIN_IMG_PATH, TEST_IMG_PATH, df_train, df_test)
        cropped.save(os.path.join(Dataset_CroppedPath[0], row['img_file']))
        printProgress(i, total_df_train, prefix=row['img_file'], suffix='done.', decimals=1, barLength=100)
    print("\nSuccessfully cropped the entire Train image.")

    # Process Test Image Crop
    for i, row in df_test.iterrows():
        cropped = crop_boxing_img(row['img_file'], TRAIN_IMG_PATH, TEST_IMG_PATH, df_train, df_test)
        cropped.save(os.path.join(Dataset_CroppedPath[1], row['img_file']))
        printProgress(i, total_df_test, prefix=row['img_file'], suffix='done.', decimals=1, barLength=100)
    print("\nSuccessfully cropped the entire Test image.")

ImgCrop(parent_list, TRAIN_IMG_PATH, TEST_IMG_PATH, df_train, df_test)
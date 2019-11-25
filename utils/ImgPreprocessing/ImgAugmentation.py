# Keras API를 활용해 Image Preprocessing(Standardize, Augmentation)을 수행하는 코드
# Code Written by Myeong-Gyu.Lee
import os
import numpy as np
from ImgPreprocessing.Utils_code import printProgress
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

AugmentCount = 4 # 한장당 Augmentation할 이미지 장수
mean = np.array([144.62598745, 132.1989693, 119.10957842], dtype=np.float32).reshape((1, 1, 3)) / 255.0 # Imagenet mean
std = np.array([5.71350834, 7.67297079, 8.68071288], dtype=np.float32).reshape((1, 1, 3)) / 255.0 # Imagenet std

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../../")
parent_list = (os.listdir(parent_dir))
DatasetPath = os.path.join(parent_dir, '2019-3rd-ml-month-with-kakr')

# 이미지 폴더 경로
TRAIN_IMG_PATH = os.path.join(DatasetPath, 'train_resized')
TEST_IMG_PATH = os.path.join(DatasetPath, 'test_resized')

def ImgAugmentation(TRAIN_IMG_PATH, TEST_IMG_PATH, ImgFolderpath, AugmentCount):
    if '2019-3rd-ml-month-with-kakr' in parent_list:
        # 이미지 폴더 경로 설정 (Dataset_ResizedPath[0] : TrainSet, Dataset_ResizedPath[1] : TestSet)
        Dataset_Path = [TRAIN_IMG_PATH, TEST_IMG_PATH]
        Dataset_AugmentedPath = [os.path.join(ImgFolderpath, 'train_augmented'), os.path.join(ImgFolderpath, 'test_augmented')]

        # Augment Images
        datagen = ImageDataGenerator(
            rescale=1. / 255,  # Zero-centering
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            featurewise_center=True,  # Z-score standardize
            featurewise_std_normalization=True
        )
        datagen.mean = mean
        datagen.std = std

        for i in range(len(Dataset_AugmentedPath)):
            try:  # output_path 디렉토리 존재 여부 확인 후 없으면 디렉토리 생성(makedirs)
                if not (os.path.isdir(Dataset_AugmentedPath[i])):  # 디렉토리 존재여부 확인
                    os.makedirs(Dataset_AugmentedPath[i])
                    print("New directory '" + str(Dataset_AugmentedPath[i]) + "' has been created.")
            except OSError as e:
                print("Failed to create directory!!!!!")
                raise NotImplementedError
            CurrentImgCount = 0  # 현재 이미지 장수
            entirefilelen = len(os.listdir(Dataset_Path[i])) * int(AugmentCount)

            for root, dirs, files in os.walk(Dataset_Path[i]):
                if not files:
                    continue
                for filename in files:
                    img_path = os.path.join(root, filename)
                    img = load_img(img_path) # PIL image
                    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                    itr = datagen.flow(x, batch_size=1, save_to_dir=Dataset_AugmentedPath[i], save_prefix=os.path.splitext(filename)[0], save_format='jpg')
                    for j in range(AugmentCount):
                        a = itr.next()
                        # PreprocessandSaveImgs2h5(int(i), H5filepath=H5filename, ndarray_data=a, imgsize=img_size)
                        CurrentImgCount += 1

                    if(CurrentImgCount == entirefilelen):
                        printProgress(CurrentImgCount, entirefilelen, str(CurrentImgCount) + ' of ' + str(entirefilelen),'All files has been preprocessed!', 1, 50)
                    else:
                        printProgress(CurrentImgCount, entirefilelen, str(CurrentImgCount) + ' of ' + str(entirefilelen), filename + ' has been preprocessed!', 1, 50)

ImgAugmentation(TRAIN_IMG_PATH, TEST_IMG_PATH, DatasetPath, AugmentCount)
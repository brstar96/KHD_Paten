# 이미지 리사이즈(224*224)를 수행하는 코드
# Code Written by Myeong-Gyu.Lee
import os, cv2
from ImgPreprocessing.Utils_code import printProgress

img_size = (224, 224)

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../../")
parent_list = (os.listdir(parent_dir))
DatasetPath = os.path.join(parent_dir, '2019-3rd-ml-month-with-kakr')

# 이미지 폴더 경로
TRAIN_IMG_PATH = os.path.join(DatasetPath, 'train')
TEST_IMG_PATH = os.path.join(DatasetPath, 'test')

def ImgResize(img_size, TRAIN_IMG_PATH, TEST_IMG_PATH, ImgFolderpath, parent_list):
    if '2019-3rd-ml-month-with-kakr' in parent_list:
        # 이미지 폴더 경로 설정 (Dataset_ResizedPath[0] : TrainSet, Dataset_ResizedPath[1] : TestSet)
        Dataset_Path = [TRAIN_IMG_PATH, TEST_IMG_PATH]
        Dataset_ResizedPath = [os.path.join(ImgFolderpath, 'train_resized'), os.path.join(ImgFolderpath, 'test_resized')]

        # Resize Images
        for i in range(len(Dataset_ResizedPath)):
            try:  # output_path 디렉토리 존재 여부 확인 후 없으면 디렉토리 생성(makedirs)
                if not (os.path.isdir(Dataset_ResizedPath[i])):  # 디렉토리 존재여부 확인
                    os.makedirs(Dataset_ResizedPath[i])
                    print("New directory '" + str(Dataset_ResizedPath[i]) + "' has been created.")
            except OSError as e:
                print("Failed to create directory!!!!!")
                raise NotImplementedError
            entirefilelen = len(os.listdir(Dataset_Path[i]))

            for root, dirs, files in os.walk(Dataset_Path[i]):
                if not files:
                    continue
                CurrentImgCount = 1  # 현재 이미지 장수
                for filename in files:
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path, 1)  # RGB 3채널 이미지로 이미지 read
                    img = cv2.resize(img, img_size)  # img_size로 리사이즈
                    r, g, b = cv2.split(img)  # r, g, b 채널로 split
                    img = cv2.merge([r, g, b])  # Standardize된 채널을 r, g, b 순서로 merge

                    cv2.imwrite(Dataset_ResizedPath[i] + "/" + filename, img)

                    if (CurrentImgCount== entirefilelen):
                        printProgress(CurrentImgCount, entirefilelen, str(CurrentImgCount) + ' of ' + str(entirefilelen), 'All files has been resized to ' + str(img_size)+ '!', 1, 50)
                    else:
                        printProgress(CurrentImgCount, entirefilelen, str(CurrentImgCount) + ' of ' + str(entirefilelen), filename + ' has been resized to ' + str(img_size)+ '!', 1, 50)
                    CurrentImgCount += 1
    else:
        print('There is no images to resize.')
        print('Go https://www.kaggle.com/c/14724/download-all and download & unzip dataset to ' + parent_dir + '.')
        raise NotImplementedError

ImgResize(img_size, TRAIN_IMG_PATH, TEST_IMG_PATH, DatasetPath, parent_list)
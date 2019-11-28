# Check anomaly images(이미지 채널 개수가 특정 개수가 아닌 파일을 찾아 출력)
import cv2, os
from tqdm import tqdm

dataset_img_savepath = '../../2019-3rd-ml-month-with-kakr/test'

# data path
dataset_img_savepath_list = os.listdir(dataset_img_savepath)

# .jpg파일만 조회
file_list_jpg = [file for file in dataset_img_savepath_list if file.endswith(".jpg")]
print ("file_list_jpg : {}".format(len(file_list_jpg)))

tbar = tqdm(range(len(file_list_jpg)), desc='\r')

for i in tbar:
    # get path of original and category images
    img_image_path = os.path.join(dataset_img_savepath, dataset_img_savepath_list[i])

    # open original and category images
    img_image = cv2.imread(img_image_path, cv2.IMREAD_COLOR)
    img_c, img_w, img_h = img_image.shape

    if img_h == 3:
        continue
    else:
        print('This original image is broken : ', img_image_path)
print('Whole images are passed. No images are broken.')

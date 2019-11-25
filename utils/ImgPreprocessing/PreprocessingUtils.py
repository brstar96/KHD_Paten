# File type 변환, .h5파일 인코딩/디코딩을 위한 코드입니다.
# Code Written by Myeong-Gyu.Lee
import h5py, os, cv2,sys
import numpy as np
from glob import glob
from keras.preprocessing.image import load_img, img_to_array

imgdata_path = os.path.join(os.getcwd(), 'ResizedImgs')
H5filename = os.path.join(os.getcwd(), 'PreprocessedImgs.h5')
imagesPath = glob(os.path.join(imgdata_path, "*.jpg"))
img_size = (224, 224, 3)

def ImageLoader(data_path):
    # Create empty list for img datas.
    img_list = []

    # loading Imgs from directory and convert to numpy array.
    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = load_img(img_path) # PIL 이미지 객체
            except:
                continue
            img_list.append(img)
    return img_list

def ImgConvert2ndarray(imgFile):
    # ImgFiles = ImageLoader(ImgDataPath)
    # print(len(ImgFiles))
    # NdArr = []
    # for i in range(len(ImgFiles)):
    #     NdArr.append(img_to_array(ImgFiles[i]))
    # print(len(NdArr))
    # return NdArr # list[ndarray[1], ndarray[2], ... ]
    return img_to_array(imgFile)

def PreprocessandSaveImgs2h5(H5filepath, ndarray_data, imgsize):
    with h5py.File(H5filepath, 'w') as hf:
        ndarray_data = ndarray_data[0]

        # b, g, r = cv2.split(ndarray_data)
        # rgb_img = cv2.merge([r, g, b])
        # rgb_img = cv2.resize(rgb_img, (imgsize[0], imgsize[1]))
        hf.create_dataset(
            # name=str(int(i)),
            name='test',
            data=ndarray_data,
            # shape= imgsize,
            # maxshape= imgsize,
            # compression="gzip",
            # compression_opts=9
            )
        print(".h5 file has been created.")


def Loadh5(H5filepath):
    with h5py.File(H5filepath, 'r') as hf:
        img = hf["data1"].value
        cv2.imwrite('.' + "/" + 'test.jpg', img)


#
# LoadedImgs = ImgConvert2ndarray(imgdata_path, img_size) # list[ndarray[1], ndarray[2], ... ]
# x_train = np.asarray(LoadedImgs, dtype=np.float32)  # (25, 224, 224, 3)
# PreprocessandSaveImgs2h5(H5filename, x_train, img_size)  # .h5파일로 ndarray들을 저장
# Loadh5(H5filename) # WIP


#
hf = h5py.File(H5filename, 'w')
ImgFiles = ImageLoader(imgdata_path)
for k in range(len(ImgFiles)):
    print(k,"finished",round((k+1)/len(ImgFiles), 3)*100,"% finished")
    toSave = ImgConvert2ndarray(ImgFiles[k])
    b, g, r = cv2.split(toSave)
    rgb_img = cv2.merge([r, g, b])
    # rgb_img = cv2.resize(rgb_img, (img_size[0], img_size[1]))
    hf.create_dataset(
        name='data'+str(k),
        data=rgb_img,
    )

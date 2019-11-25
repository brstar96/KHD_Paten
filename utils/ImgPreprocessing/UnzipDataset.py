# Code Written by Myeong-Gyu.Lee
import os, zipfile

def unzip(parent_list):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../../")

    if '2019-3rd-ml-month-with-kakr' in parent_list:
        DatasetPath = os.path.join(parent_dir, '2019-3rd-ml-month-with-kakr')
        print('Current Dataset Path : ' + DatasetPath)

        file_list_zip = [file for file in os.listdir(DatasetPath) if file.endswith(".zip")]

        for i in range(len(file_list_zip)):
            NewFolderpath = os.path.join(DatasetPath, os.path.splitext(file_list_zip[i])[0])

            if not (os.path.isdir(NewFolderpath)):  # 디렉토리 존재여부 확인
                os.makedirs(NewFolderpath)
                print("New directory '" + str(NewFolderpath) + "' has been created.")

            print('Now Unzip :', NewFolderpath)
            zip = zipfile.ZipFile(os.path.join(DatasetPath, file_list_zip[i]))
            zip.extractall(NewFolderpath)
            zip.close()
            print('Done :', NewFolderpath)

            if os.path.isfile(os.path.join(DatasetPath, file_list_zip[i])):
                os.remove(os.path.join(DatasetPath, file_list_zip[i]))
                print(file_list_zip[i] + ' has been removed.')
    else:
        print('Go https://www.kaggle.com/c/14724/download-all and download & unzip dataset to ' + parent_dir + '.')
        raise NotImplementedError
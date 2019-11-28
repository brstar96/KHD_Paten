import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from torchvision import datasets, transforms

# def calcMeanStd(dataloader):
#     tbar = tqdm(dataloader)
#     pop_mean = []  # 배치 단위로 이미지 픽셀 평균을 담을 리스트
#     pop_std0 = []  # 배치 단위로 이미지 픽셀 표준편차를 담을 리스트
#     pop_std1 = []  # 배치 단위로 이미지 픽셀 표준편차를 담을 리스트 2
#
#     for i, data in enumerate(tbar, 0):
#         # shape (batch_size, 3, height, width)
#         numpy_image = data[0].numpy()
#         # shape (3,)
#         batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
#         batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
#         batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)  # ddof : 비편향 분산
#
#         print("Batch mean : ", batch_mean)
#         print("Batch std0/1 : ", batch_std0, batch_std1)
#
#         pop_mean.append(batch_mean) # 각 배치에 대해 구한 픽셀값들의 mean
#         pop_std0.append(batch_std0) # 각 배치에 대해 구한 픽셀값들의 std값들
#         pop_std1.append(batch_std1) # 각 배치에 대해 비편향 분산을 이용해 구한 픽셀값들의 std값들
#         print("Batch pop_mean : ", pop_mean)
#         print("Batch pop_std0/1 : ", pop_std0, pop_std1)
#
#     # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
#     mean = np.array(pop_mean).mean(axis=0)
#     std0 = np.array(pop_std0).mean(axis=0)
#     std1 = np.array(pop_std1).mean(axis=0)
#
#     return mean, std0, std1

def calcMeanStd(dataloader):
    """Compute the mean and sd in an online fashion
        Var[x] = E[X^2] - E^2[X]
    """
    tbar = tqdm(dataloader)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tbar:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def main():
    # Build pytorch dataset and batch dataloader
    batch_size = 500
    dataset = datasets.ImageFolder(root='../../2019-3rd-ml-month-with-kakr/',
                                   transform=transforms.Compose([
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor()]))
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    print("\nlength of dataloader : ", len(dataloader) * batch_size)
    entire_mean, entire_std = calcMeanStd(dataloader)
    print("Entire dataset`s mean and std(mean/std0/std1) : ", entire_mean, entire_std)

if __name__ == "__main__":
    main()

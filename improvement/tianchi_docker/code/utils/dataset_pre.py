import pandas as pd
import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader_pre(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path_precip = os.path.join(data_path, 'Precip/')
        self.imgs_path_radar = os.path.join(data_path, 'Radar/')
        self.imgs_path_wind = os.path.join(data_path, 'Wind/')

    def get_image(self, index):
        # 读取index
        item = os.listdir(self.data_path + 'Precip/')
        item = item[index]
        item_precip = os.listdir(self.data_path + 'Precip/' + item)
        item_radar = os.listdir(self.data_path + 'Radar/' + item)
        item_wind = os.listdir(self.data_path + 'Wind/' + item)
        for i in range(20):
            # 根据index读取图片
            image_path_precip = self.imgs_path_precip + item + '/' + str(item_precip[i])
            image_path_radar = self.imgs_path_radar + item + '/' + str(item_radar[i])
            image_path_wind = self.imgs_path_wind + item + '/' + str(item_wind[i])

            # 读取训练图片和标签图片
            image_precip = cv2.imread(image_path_precip, cv2.IMREAD_UNCHANGED)
            image_radar = cv2.imread(image_path_radar, cv2.IMREAD_UNCHANGED)
            image_wind = cv2.imread(image_path_wind, cv2.IMREAD_UNCHANGED)
            # 将数据转为单通道的图片
            image_precip = image_precip.reshape(1, image_precip.shape[0], image_precip.shape[1])
            image_radar = image_radar.reshape(1, image_radar.shape[0], image_radar.shape[1])
            image_wind = image_wind.reshape(1, image_wind.shape[0], image_wind.shape[1])
            image_all = np.concatenate((image_precip, image_radar, image_wind))
            if i == 0:
                image = image_all
            else:
                image = np.concatenate((image, image_all))

        return image, item

    def __getitem__(self, index):
        return self.get_image(index)

    def __len__(self):
        # 返回训练集大小
        return len(os.listdir(self.data_path + 'Precip'))


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader_pre("../TestB1/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=False)
    for image, label in train_loader:
        print(image.shape)

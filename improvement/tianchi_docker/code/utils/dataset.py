import pandas as pd
import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path, csv_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.csv_path = csv_path
        self.imgs_path_precip = os.path.join(data_path, 'Precip/precip')
        self.imgs_path_radar = os.path.join(data_path, 'Radar/radar')
        self.imgs_path_wind = os.path.join(data_path, 'Wind/wind')

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def get_image(self, index):
        # 读取index
        csv = pd.read_csv(self.csv_path, header=None)
        item_image = csv.iloc[index, :20]
        item_label = csv.iloc[index, -20:]

        for i in range(20):
            # 根据index读取图片
            image_path_precip = self.imgs_path_precip + '_' + str(item_image[i])
            image_path_radar = self.imgs_path_radar + '_' + str(item_image[i])
            image_path_wind = self.imgs_path_wind + '_' + str(item_image[i])
            # 根据image_path生成label_path
            label_path_precip = self.imgs_path_precip + '_' + str(item_label[i + 20])
            label_path_radar = self.imgs_path_radar + '_' + str(item_label[i + 20])
            label_path_wind = self.imgs_path_wind + '_' + str(item_label[i + 20])

            # 读取训练图片和标签图片
            image_precip = cv2.imread(image_path_precip, cv2.IMREAD_UNCHANGED)
            image_radar = cv2.imread(image_path_radar, cv2.IMREAD_UNCHANGED)
            image_wind = cv2.imread(image_path_wind, cv2.IMREAD_UNCHANGED)
            label_precip = cv2.imread(label_path_precip, cv2.IMREAD_UNCHANGED)
            label_radar = cv2.imread(label_path_radar, cv2.IMREAD_UNCHANGED)
            label_wind = cv2.imread(label_path_wind, cv2.IMREAD_UNCHANGED)

            image_precip = image_precip.reshape(1, image_precip.shape[0], image_precip.shape[1])
            image_radar = image_radar.reshape(1, image_radar.shape[0], image_radar.shape[1])
            image_wind = image_wind.reshape(1, image_wind.shape[0], image_wind.shape[1])
            label_precip = label_precip.reshape(1, label_precip.shape[0], label_precip.shape[1])
            label_radar = label_radar.reshape(1, label_radar.shape[0], label_radar.shape[1])
            label_wind = label_wind.reshape(1, label_wind.shape[0], label_wind.shape[1])
            # 处理标签，将像素值为255的改为1
            if label_precip.max() > 1:
                label_precip = label_precip / 255
            if label_radar.max() > 1:
                label_radar = label_radar / 255
            if label_wind.max() > 1:
                label_wind = label_wind / 255

            image_all = np.concatenate((image_precip, image_radar, image_wind))
            label_all = np.concatenate((label_precip, label_radar, label_wind))
            if i == 0:
                image = image_all
                label = label_all
            else:
                image = np.concatenate((image, image_all))
                label = np.concatenate((label, label_all))

        return image, label

    def __getitem__(self, index):
        return self.get_image(index)

    def __len__(self):
        # 返回训练集大小
        csv = pd.read_csv(self.csv_path, header=None)
        return csv.shape[0]


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("../Train/", "../Train.csv")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)

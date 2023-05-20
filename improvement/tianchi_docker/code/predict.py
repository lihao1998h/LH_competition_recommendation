import glob
import random

import numpy as np
import torch
import os
import cv2
import torch.nn as nn
from model.unet_model import UNet
from utils.dataset_pre import ISBI_Loader


def pred_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    val_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练模式
    net.eval()
    for image, item in val_loader:
        # 将数据拷贝到device中
        image = image.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        pred = net(image)
        # 提取结果
        pred = np.array(pred.data.cpu())[0]
        pred = pred * 255
        # 处理结果
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        # 保存图片
        save_path_precip = '../submit/' + 'pre/' + 'Precip/' + item[0]
        save_path_radar = '../submit/' + 'pre/' + 'Radar/' + item[0]
        save_path_wind = '../submit/' + 'pre/' + 'Wind/' + item[0]
        if not os.path.exists(save_path_precip):
            os.makedirs(save_path_precip)
            os.makedirs(save_path_radar)
            os.makedirs(save_path_wind)
        for i in range(pred.shape[0]):
            if i % 3 == 0:
                cv2.imwrite(save_path_precip + '/precip_' + str(i // 3 + 1).zfill(3) + '.png', pred[i])
            elif i % 3 == 1:
                cv2.imwrite(save_path_radar + '/radar_' + str(i // 3 + 1).zfill(3) + '.png', pred[i])
            elif i % 3 == 2:
                cv2.imwrite(save_path_wind + '/wind_' + str(i // 3 + 1).zfill(3) + '.png', pred[i])


def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_torch()
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=60, n_classes=60)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('../user_data/model_data/best_model.pth', map_location=device))
    # 指定训练集地址，开始验证
    data_path = "../data/TestB3_20/"
    pred_net(net, device, data_path)


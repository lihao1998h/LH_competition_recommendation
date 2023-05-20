import os
import random

import numpy as np

from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch


def test_net(net, device, data_path, data_csv, epochs=1, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path, data_csv)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.eval()
        # 按照batch_size开始训练
        for image, label in train_loader:
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())


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
    net.load_state_dict(torch.load('./best_model.pth', map_location=device))
    # 指定训练集地址，开始训练
    data_path = "./TestA/"
    data_csv = "./TestA.csv"
    test_net(net, device, data_path, data_csv)

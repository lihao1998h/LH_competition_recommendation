import os
import random

import numpy as np
import cv2
from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from utils.dataset_pre import ISBI_Loader_pre
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ai_hub import inferServer
import json
import os
import shutil
import base64
from io import BytesIO
from PIL import Image

RECORDS_NUM = 21
INPUT_PNG_NUM = 20
OUTPUT_PNG_NUM = 20


def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_net(net, device, data_path, data_csv, epochs=20, batch_size=4, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path, data_csv)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 定义Loss算法
    criterion = nn.MSELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练

        for i, (image, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), '../user_data/model_data/best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()


def pred_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader_pre(data_path)
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
        save_path_precip = '../submit/' + 'Precip/' + item[0]
        save_path_radar = '../submit/' + 'Radar/' + item[0]
        save_path_wind = '../submit/' + 'Wind/' + item[0]
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


def evalset():
    data_path = "./submit"  # 数据目录，选手替换为本地实际路径 ./eval_data

    for category in ['Wind', 'Precip', 'Radar']:
        for png_idx in range(1, INPUT_PNG_NUM + 1):
            file_name = os.path.join(data_path, category, f"{category.lower()}_{png_idx:03d}.png")

            image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            image_ = image.reshape(1, image.shape[0], image.shape[1])

            # binary_content = open(file_name, 'rb').read()
            # base64_bytes = base64.b64encode(binary_content)
            # base64_string = base64_bytes.decode('utf-8')
            # sub_file_name = file_name.split('/')[-1]
            if category == 'Wind' and png_idx == 1:
                output = image_
            else:
                output = np.concatenate((image_, output))
    output = output[np.newaxis, :, :, :]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.FloatTensor(output).to(device)


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        print("init_myInfer")

    # 数据输入elem: dict结构,表示某一特定时刻的Wind/Radar/Precip数据
    # elem["Wind"]: {
    #   "wind_001.png": "base64的png图片内容",
    #   "wind_002.png": "base64的png图片内容",
    #   ...
    #   "wind_020.png": "base64的png图片内容",
    # }
    # elem["Radar"]: {
    #   "radar_001.png": "base64的png图片内容",
    #   "radar_002.png": "base64的png图片内容",
    #   ...
    #   "radar_020.png": "base64的png图片内容",
    # }
    # elem["Precip"]: {
    #   "precip_001.png": "base64的png图片内容",
    #   "precip_002.png": "base64的png图片内容",
    #   ...
    #   "precip_020.png": "base64的png图片内容",
    # }

    # 数据前处理
    # 本示例的做法是将天池服务器传入的特定时刻的Wind/Radar/Precip数据写入到本地图片，供选手参考。
    # 选手可以根据自己的处理逻辑进行数据预处理后return给pridect(data)。
    def pre_process(self, data):
        print("pre_process")
        data = data.get_data()

        # json process
        json_data = json.loads(data.decode('utf-8'))

        # 将图片写到本地的submit目录：
        if os.path.exists('submit'):
            shutil.rmtree('submit')
            print('Delete submit folder')
            os.makedirs('submit')
            print('Create submit folder')

        for category in ['Wind', 'Precip', 'Radar']:
            category_path = os.path.join('submit', category)
            os.makedirs(category_path)
            for png_name, base64_string in json_data[category].items():
                ## 请选手注意，在天池的流评测服务器环境下，此处获取到的base64_string其实是一个list结构，非str，选手需要显性得取第一个元素才能得到正确的图片base64编码内容
                base64_string = base64_string[0]
                file_name = os.path.join(category_path, png_name)
                img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_string)))
                img.save(file_name)
                # print ('success save ', file_name)

        eval_data = evalset()

        return eval_data

    # 模型预测：默认执行self.model(preprocess_data)，一般不用重写
    # 如需自定义，可覆盖重写
    def predict(self, data):
        ret = self.model(data)
        return ret

    # 数据后处理
    def post_process(self, pred):
        print("post_process")

        # 此处示例仅演示生成返回格式，未经过predict（data）;直接读取本地图片。选手可根据实际逻辑实现。
        # 与pre_process类似，将模型预测后的图片文件，以base64的方式返回，返回字段见如下的定义：
        # elem["Wind"]: {
        #   "wind_001.png": "base64的png图片内容",
        #   "wind_002.png": "base64的png图片内容",
        #   ...
        #   "wind_020.png": "base64的png图片内容",
        # }
        # elem["Radar"]: {
        #   "radar_001.png": "base64的png图片内容",
        #   "radar_002.png": "base64的png图片内容",
        #   ...
        #   "radar_020.png": "base64的png图片内容",
        # }
        # elem["Precip"]: {
        #   "precip_001.png": "base64的png图片内容",
        #   "precip_002.png": "base64的png图片内容",
        #   ...
        #   "precip_020.png": "base64的png图片内容",
        # }

        # 提取结果
        pred = np.array(pred.data.cpu())[0]
        pred = pred * 255
        # 处理结果
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        # 保存图片
        save_path_precip = './pred/' + 'Precip/'
        save_path_radar = './pred/' + 'Radar/'
        save_path_wind = './pred/' + 'Wind/'
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
        # 导出json
        elem = {}
        # 读取模型预测好的文件，按照如下格式返回, 本例为了说明问题，直接去读输入文件的第一条作为输出结果：
        elem['Wind'] = {}
        elem['Radar'] = {}
        elem['Precip'] = {}
        for category in ['Wind', 'Precip', 'Radar']:
            for idx in range(1, 21):
                mock_file_path = os.path.join('pred', category, f"{category.lower()}_{idx:03d}.png")
                print('mock_file_path: ', mock_file_path)

                binary_content = open(mock_file_path, 'rb').read()
                base64_bytes = base64.b64encode(binary_content)
                base64_string = base64_bytes.decode('utf-8')

                file_name = os.path.join('', f"{category.lower()}_{idx:03d}.png")
                print('Post_process: ', category, file_name)
                elem[category][file_name] = base64_string

        # 返回json格式
        return json.dumps(elem)


if __name__ == "__main__":
    seed_torch()
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=60, n_classes=60)
    # 将网络拷贝到deivce中
    net.to(device=device)
    net.load_state_dict(torch.load('./user_data/model_data/best_model.pth', map_location=device))

    my_infer = myInfer(net)
    my_infer.run(debuge=True)  # , nohug=False)  # 默认为("127.0.0.1", 80)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息
    # ！！！nohug参数默认为False,第二阶段记得修改为True提交，即可持久化inferServer服务，等待正式测评

    # 指定训练集地址，开始训练
    # data_dir = '/tcdata'
    # data_path = data_dir + "/Train/"
    # data_csv = data_dir + "/Train.csv"
    # train_net(net, device, data_path, data_csv)
    # predict
    # net.load_state_dict(torch.load('./user_data/model_data/best_model.pth', map_location=device))
    # 指定训练集地址，开始验证
    # data_path = data_dir + "/TestB3_20/"
    # pred_net(net, device, data_path)

#精简版示例代码：
from torch.utils.data import Dataset, DataLoader
import time
import requests
import json
import os
import shutil
import base64
from io import BytesIO
from PIL import Image
import torch
from unet_model import UNet
import numpy as np

import cv2
INDEX_URL = "http://127.0.0.1:8080/tccapi"

# 每天有21条记录，正式比赛期间可能不固定
RECORDS_NUM = 21
INPUT_PNG_NUM = 20
OUTPUT_PNG_NUM = 20

# 每天数据格式如下：
# 解压后的一级目录有3个: Wind, Radar, Precip
# 每个一级目录下面分21个子目录，从001到021编号，对应21次流评测
# 每个子目录下面有20张图片，编号从001到020，以Wind为例，文件命名是wind_001.png, wind_002.png, ...wind_020.png

# 喂给选手模型的每条数据的格式定义为一个dict：
# elem["index"]: 流评测idx
# elem["Wind"]: {
#   "wind_001.png": "二进制的png图片内容",
#   "wind_002.png": "二进制的png图片内容",
#   ...
#   "wind_020.png": "二进制的png图片内容",
# }
# elem["Radar"]: {
#   "radar_001.png": "二进制的png图片内容",
#   "radar_002.png": "二进制的png图片内容",
#   ...
#   "radar_020.png": "二进制的png图片内容",
# }
# elem["Precip"]: {
#   "precip_001.png": "二进制的png图片内容",
#   "precip_002.png": "二进制的png图片内容",
#   ...
#   "precip_020.png": "二进制的png图片内容",
# }

## 相应的，要求选手返回的格式同 输入格式，也按照这4个字段来返回

class evalset(Dataset):
    def __init__(self):
        self.data = []
        data_path = "/data1/lihao/competitions/weather/TestB1"  # 数据目录，选手替换为本地实际路径 ./eval_data

        try:
            for record_idx in range(1, RECORDS_NUM + 1):
                elem = {}
                elem["index"] = str(record_idx)
                elem['Wind'] = {}
                elem['Precip'] = {}
                elem['Radar'] = {}
                for category in ['Wind', 'Precip', 'Radar']:
                    for png_idx in range(1, INPUT_PNG_NUM + 1):
                        file_name = os.path.join(data_path, category, f"{record_idx:03d}", f"{category.lower()}_{png_idx:03d}.png")
                        binary_content = open(file_name, 'rb').read()
                        base64_bytes = base64.b64encode(binary_content)
                        base64_string = base64_bytes.decode('utf-8')
                        sub_file_name = file_name.split('/')[-1]
                        elem[category][sub_file_name] = base64_string
                #print('Evalset init: index ', record_idx, ' type: ', type(elem))
                self.data.append(elem)
                # for record_idx in range(1, RECORDS_NUM + 1):
                #     elem = {}
                #     elem["index"] = str(record_idx)
                #     elem['Wind'] = {}
                #     elem['Precip'] = {}
                #     elem['Radar'] = {}
                #     for category in ['Wind', 'Precip', 'Radar']:
                #         for png_idx in range(1, INPUT_PNG_NUM + 1):
                #             file_name = os.path.join(data_path, category, f"{record_idx:03d}",
                #                                      f"{category.lower()}_{png_idx:03d}.png")
                #             binary_content = open(file_name, 'rb').read()
                #             base64_bytes = base64.b64encode(binary_content)
                #             base64_string = base64_bytes.decode('utf-8')
                #             sub_file_name = file_name.split('/')[-1]
                #             elem[category][sub_file_name] = base64_string
                #     # print('Evalset init: index ', record_idx, ' type: ', type(elem))
                #     self.data.append(elem)
        except Exception as e:
            print(e)
            self.data = []

    def __getitem__(self, index):
        signl_data = self.data[index]
        return signl_data

    def __len__(self):
        return len(self.data)


def loader2json(data):
    send_json = data
    return send_json


def evalset_local():
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

def pre_process(data):
    print("pre_process")

    # 将图片写到本地的submit目录：
    if os.path.exists('submit'):
        shutil.rmtree('submit')
        print('Delete submit folder')
        os.makedirs('submit')
        print('Create submit folder')

    for category in ['Wind', 'Precip', 'Radar']:
        category_path = os.path.join('submit', category)
        os.makedirs(category_path)
        for png_name, base64_string in data[category].items():
            ## 请选手注意，在天池的流评测服务器环境下，此处获取到的base64_string其实是一个list结构，非str，选手需要显性得取第一个元素才能得到正确的图片base64编码内容
            base64_string = base64_string[0]
            file_name = os.path.join(category_path, png_name)
            img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_string)))
            img.save(file_name)
            # print ('success save ', file_name)

    eval_data = evalset_local()

    return eval_data


def local_eval(data):
    input = pre_process(data)
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=60, n_classes=60)
    # 将网络拷贝到deivce中
    net.to(device=device)
    net.load_state_dict(torch.load('./user_data/model_data/best_model.pth', map_location=device))

    ret = net(input)
    return ret


def eval_dataloader(evalloader, log=True):
    store_data = {}
    print("eval_start")

    for i, data in enumerate(evalloader):
        data_json = loader2json(data)
        index_str = data_json['index'][0]
        ret = local_eval(data_json)
        # res, cost_time = send_eval(data_json, log)
        # if res != None:
        #     res_elem = json.loads(res)
        #     res_elem['index'] = index_str
        #     store_data[index_str] = res_elem
    # store(store_data)
    print("eval_end")


# 返回数据做解析
def analysis_res(res):
    return res

# 将选手返回的结果写到当前目录的3个一级目录（Wind、Radar、Precip），并输出成最终的压缩格式。
def store(data):
    # 检查三个一级目录是否存在，存在的话先删除再建
    for category in ['Wind', 'Precip', 'Radar']:
        if os.path.exists(category):
            shutil.rmtree(category)
            print('Delete: ', category)
        os.mkdir(category)
        print('Create: ', category)

    # 为每条流评测数据选手预测的结果，写到对应的文件目录中：
    for key, elem in data.items():
        record_idx = int(elem['index'])
        for category in ['Wind', 'Precip', 'Radar']:
            category_path = os.path.join(category, f"{record_idx:03d}")
            os.mkdir(category_path)
            #print(elem[category])
            for png_name, base64_string in elem[category].items():
                file_name = os.path.join(category_path, png_name)
                print (file_name)
                img = Image.open(BytesIO(base64.urlsafe_b64decode(base64_string)))
                img.save(file_name)

    # 最后一步，压缩成zip文件，再去调用评测脚本，注意：这个比赛的评测程序要求必须是一天的数据都收集全了，才能算分
    os.system('zip -r result.zip Wind Precip Radar')

def send_eval(data_json, log):
    url = "http://127.0.0.1:8080/tccapi"
    start = time.time()
    # try:
    #     data = json.dumps(data_json)
    #     res = requests.post(url=url, data=data, timeout=60)
    #     res_batch = json.loads(res.text)
    #     res = json.dumps(res_batch)
    #
    #     cost_time = time.time() - start
    #     res = analysis_res(res)
    #     return res, cost_time
    # except Exception as e:
    #     print("request faild:")
    #     print(e)
    #     print("xxxxx")
        #res = json.dumps(res_batch)

    data = json.dumps(data_json)
    res = requests.post(url=url, data=data, timeout=60) #调试阶段超时60秒，正式阶段待调整
    if res.status_code == 200:
        res_batch = json.loads(res.text)
        res = json.dumps(res_batch)
    else:
        res = None

    cost_time = time.time() - start
    res = analysis_res(res)
    return res, cost_time



def open_eval():
    print('server started !')
    # server started callback
    try:
        eval_data = evalset()
        evalloader = DataLoader(eval_data, batch_size=1, shuffle=False)
        eval_dataloader(evalloader)
    except Exception as e:
        print(e)
        # report_exception(str(e))
    # requests.post(INDEX_URL, "exit")


import threading
from urllib import request

if __name__ == "__main__":
    threading.Thread(target=open_eval).start()

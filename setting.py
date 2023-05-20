import argparse
import numpy as np
# import tensorflow as tf
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import torch


parser = argparse.ArgumentParser()
# 通用参数
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--task_type', type=str, default='recommendation',
                    choices=['ctr', 'recommendation', 'recommendation_multimodal'])
parser.add_argument('--match_num', type=int, default=100, help='only for recommendation')


parser.add_argument('--data_root', type=str, default='D:/data/KDD2023')
# 'D:/data/KDD2023'
# '/data1/lihao/competitions/KDD2023'
parser.add_argument('--output_root', type=str, default='./output')


# dataloader parser
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)




# EDA parser
parser.add_argument('--EDA', type=bool, default=False)
parser.add_argument('--Auto_EDA', type=bool, default=True)
parser.add_argument('--is_reduce_mem_usage', type=bool, default=False)  # 当数据量过大时需要减少内存占用

# Feature Engineering parser
parser.add_argument('--FE', type=bool, default=True)

# model parser
parser.add_argument('--num_epoch', type=int, default=10,
                    help='number of training epochs')
# train parser
parser.add_argument('--model', type=str, default='NARM', choices=['lstm', 'tree', 'CNN1D', 'NN', 'NARM'])
parser.add_argument('--tree_model', type=str, default='lgb', choices=['catboost', 'lgb'])
parser.add_argument('--metrics', type=str, default='rmse', choices=['rmse', 'l1', 'mape', 'acc'])
# parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--pretrained', type=bool, default=False)

args = parser.parse_args()




# 初始设置
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not os.path.exists(args.output_root):
    os.mkdir(args.output_root)

# 1 <pandas设置>
# pd.set_option("display.max_columns",32)  # 显示最大列数
pd.options.display.max_rows = 1000  # 显示最大行数，若为None则为无穷
pd.options.display.max_columns = 20  # 显示最大列数，若为None则为无穷
# pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 小数格式设置

# 2 <warning设置>
warnings.filterwarnings("ignore")  # 过滤警告文字

# 3 <plt设置>
# %matplotlib inline # 不用show就可以显示图片，这一条需要手动在notebook中输入
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 在图片中显示中文
plt.rcParams['figure.figsize'] = (10.0, 5.0)

plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-dark')

# 4 <sns设置>

sns.set()
sns.set(font='SimHei', font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小


# 5 tensorflow设置
# tf.random.set_seed(args.seed)
# 5.1 设置gpu内存自增长
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# 6 torch设置

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
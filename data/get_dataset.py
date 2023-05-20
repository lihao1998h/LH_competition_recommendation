import time
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

def get_dataset(args):
    print('-'*20, '读取数据开始', '-'*20)
    LoadData_time_start = time.time()

    productfile = pd.read_csv(os.path.join(args.data_root, 'products_train.csv'))  # 商品信息
    userfile = pd.read_csv(os.path.join(args.data_root, 'sessions_train.csv'))  # 用户信息
    task = 'task1'
    testfile = pd.read_csv(os.path.join(args.data_root, f'sessions_test_{task}.csv'))

    LoadData_time_end = time.time()
    print('-' * 20, '读取数据完成，耗时： ', LoadData_time_end - LoadData_time_start, '秒', '-' * 20)


    # 预处理
    print('-'*20, '预处理开始', '-'*20)
    preprocess_time_start = time.time()
    ## 特征预处理
    fea_change = True
    if fea_change:
        product2idx = dict(zip(productfile['id'].unique(), range(1, productfile['id'].nunique() + 1)))
        idx2product = dict(zip(range(1, productfile['id'].nunique() + 1), productfile['id'].unique()))
        args.product_num = productfile['id'].nunique()

        locale_product_dict = dict()

        for locale in args.locales:
            locale_product_dict[locale] = [product2idx[x] for x in list(productfile[productfile['locale'] == locale]['id'].unique())]

        def str2list(x):
            x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
            l = [product2idx[i] for i in x.split() if i]
            return l

        userfile['prev_items'] = userfile['prev_items'].apply(lambda x: str2list(x))
        testfile['prev_items'] = testfile['prev_items'].apply(lambda x: str2list(x))
        userfile['next_item'] = userfile['next_item'].apply(lambda x: product2idx[x])

    # 数据集划分
    df_train, df_valid, _, _ = train_test_split(
        userfile, userfile['locale'], test_size=0.1, random_state=args.seed, stratify=userfile['locale'])

    print(f'df_train.shape = {df_train.shape}, df_valid.shape = {df_valid.shape}')
    train = (list(df_train["prev_items"]), list(df_train["next_item"]))
    valid = (list(df_valid["prev_items"]), list(df_valid["next_item"]))
    test = (list(testfile["prev_items"]), None)

    train_dataset = TrainDataset(train)
    valid_dataset = TrainDataset(valid)
    test_dataset = TrainDataset(test, test=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_train, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test)

    preprocess_time_end = time.time()
    print('-' * 20, '预处理完成，耗时： ', preprocess_time_end - preprocess_time_start, '秒', '-' * 20)

    return train_loader, val_loader, test_loader

def data_preprocess(productfile, userfile):
    # 数据预处理：合并文件、修改特征、数据填充、数据截断、数据集划分
    preprocess = False  # 如需使用预处理，请自定义
    if preprocess:
        def conbine_date(df):
            # 整合日期列
            df['date'] = df['年'].astype(str) + '-' + df['月'].map(
                lambda x: '0' + str(x) if x <= 9 else str(x)) + '-' + df['日']
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop(['年', '月', '日'], axis=1)
            return df

        df_train = conbine_date(df_train)
        df_test = conbine_date(df_test)

        # 如果有文本的列可以用labelencoder转化为数字
        # le = LabelEncoder()
        # items['family'] = le.fit_transform(items['family'].values)

        print('data are loaded')

        # 表合并
        # installed = installed.merge(userfile, how='left', on='userID')

        # 删数据
        # 对dict中标记的删到指定之前，-1的删最后20%
        # df_tmp = pd.DataFrame()
        # for id in df_train['session_id'].unique():
        #     if dict[id][2] != -1:
        #         tmp = df_train.loc[df_train['session_id'] == id].loc[df_train['rank'] <= dict[id][2]]
        #         df_tmp = pd.concat([df_tmp, tmp])
        #     else:
        #         max_rank = df_train.loc[df_train['session_id'] == id]['rank'].values.max()
        #         tmp = df_train.loc[df_train['session_id'] == id].loc[df_train['rank'] <= int(max_rank*0.8)]
        #         df_tmp = pd.concat([df_tmp, tmp])
        # df_train = df_tmp


        # 数据填充
        fill_all = False
        if fill_all:
            fill_rank = 417
            groupby = df_train.groupby(['session_id'])['rank'].max()
            sort_groupby = groupby.sort_values().reset_index()
            fill_id_list = sort_groupby[sort_groupby['rank'] < fill_rank]['session_id'].values.tolist()
            # 填充过去时间，默认用最近时刻的值填充，为了之后做特征
            for id in fill_id_list:
                max_rank = df_train[df_train['session_id'] == id]['rank'].max()
                for rank in range(max_rank+1, fill_rank):
                    fill_col = df_train[df_train['session_id'] == id][df_train['rank'] == max_rank].reset_index(drop=True)
                    fill_col['rank'] = rank
                    df_train = df_train.append(fill_col, ignore_index=True)
            df_train = df_train.sort_values(by=['session_id', 'rank']).reset_index(drop=True)


        # 构造测试集
        # start_date = "2021-12-28"
        # end_date = "2022-01-03"
        # test_date_list = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        # good_ids = df_sub_sample['商品id'].unique()
        #
        # ## 笛卡尔积
        # b = pd.DataFrame({'商品id': good_ids, 'key': [1 for _ in range(len(good_ids))]})
        # a = pd.DataFrame({'时间': test_date_list, 'key': [1 for _ in range(len(test_date_list))]})
        # df_test = b.merge(a, on='key')
        # df_test = df_test.drop('key', axis=1)
        # df_test['未来一周天均销量'] = 0
        # print('测试集构造完成')


    return df_train, df_test



def get_more_data():
    # 多文件读取
    from tqdm import tqdm
    df_all = []
    for file in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        df_all.append(df)

    bike_track = pd.concat([
        pd.read_csv(PATH + 'gxdc_gj20201221.csv'),
        pd.read_csv(PATH + 'gxdc_gj20201222.csv'),
        pd.read_csv(PATH + 'gxdc_gj20201223.csv'),
        pd.read_csv(PATH + 'gxdc_gj20201224.csv'),
        pd.read_csv(PATH + 'gxdc_gj20201225.csv')

    ])

    # 多文件
    # files = os.listdir(data_root)
    # df_train = get_data(os.path.join(data_root, files[0]))
    # for file in files[1:]:
    #     # 总文件数1449=1474-25
    #     df_temp = get_data(os.path.join(data_root, file))
    #     df_train = pd.concat((df_train, df_temp), axis=0)

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type not in [object, 'datetime64[ns]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # datetime or category
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def df_to_h5(df):
    # 从df转向hdf5，可以加快读取速度
    df.to_hdf('train_test.h5', '1.0')
    df = pd.read_hdf('train_test.h5', '1.0')



def collate_fn_train(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(hist) for hist, _ in data]
    labels = []
    padded_seq = torch.zeros(len(data), max(lens)).long()
    for i, (hist, label) in enumerate(data):
        padded_seq[i, :lens[i]] = torch.LongTensor(hist)
        labels.append(label)

    return padded_seq, torch.tensor(labels).long(), lens


def collate_fn_test(data):
    data.sort(key=lambda x: len(x), reverse=True)
    lens = [len(hist) for hist in data]
    padded_seq = torch.zeros(len(data), max(lens)).long()
    for i, hist in enumerate(data):
        padded_seq[i, :lens[i]] = torch.LongTensor(hist)

    return padded_seq, lens

class TrainDataset(Dataset):
    def __init__(self, data, test=False):
        self.data = data
        self.test = test

    def __getitem__(self, index):
        if not self.test:
            return self.data[0][index], self.data[1][index]
        else:
            return self.data[0][index]

    def __len__(self):
        return len(self.data[0])

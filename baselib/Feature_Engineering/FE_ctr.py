import pandas as pd
import numpy as np
from chinese_calendar import is_holiday
from sklearn.preprocessing import LabelEncoder
from baselib.Feature_Engineering.Feature_Engineering_base import slide_window_feature
from data.Preprocessing_base import concat_train_and_test

# 参考资料：
# https://cloud.tencent.com/developer/beta/article/1924216


def FE_user_stat(train_X, train_y, test_X, ori_feas):
    # 客视角统计特征
    ## 用户活跃小时数
    add = pd.DataFrame(data.groupby(["userID"]).hour.nunique()).reset_index()
    add.columns = ["userID", "user_active_hour"]
    data = data.merge(add, on=["userID"], how="left")

    # 构造时间窗口
    # 为每个店、每个商品使用时间窗口构造X和y
    # 有的只记录到49
    # 13-24  val 模型弄好了再去用13-24也做特征

    # 25-36 用37之后做特征
    # ...用到60试试
    # 48-59 用 60
    t_train_start = 360
    train_num = t_train_start - 25 + 1  # 24
    # t_train_start = date(2021, 11, 9)
    num_days = train_num + 1 + 1  # 训练+验证+测试
    X_l, y_l = [], []
    for i in range(num_days):
        # delta = timedelta(days=7 * i)
        delta = -1
        if i < train_num:
            X_tmp, y_tmp = slide_window_feature(train_X, train_y, t_train_start + i * delta, is_train_val=True)
            X_l.append(X_tmp)
            y_l.append(y_tmp)
        elif i == train_num:
            print('valing')
            X_val, y_val = slide_window_feature(train_X, train_y, t_train_start + (i - 1) * delta, is_train_val=True)
        elif i > train_num:
            print('testing')
            X_test = slide_window_feature(train_X, train_y, t_train_start + (i + 12 - 2) * delta, is_train_val=False)

    X_train = pd.concat(X_l, axis=0)  # 295*train_num      10325 *4324
    y_train = np.concatenate(y_l, axis=0)  # 295*train_num*12    46020 *4324
    y_val = y_val.reshape(295, 12)
    y_train = y_train.reshape(295 * train_num, 12)
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    del X_l, y_l
    return X_train, y_train, X_val, y_val, X_test


def FE_goods_stat(train_X, train_y, test_X, feas):
    # 货视角统计特征
    ## 活跃app数特征
    add = pd.DataFrame(data.groupby(["appCategory"]).appID.nunique()).reset_index()
    add.columns = ["appCategory", "appCategory_active_app"]
    data = data.merge(add, on=["appCategory"], how="left")



    class_feas = ['站号', '地区']
    time_fea = ['day', 'hour', 'month', 'year']
    train_X['label'] = train_y
    feas.append('label')
    for class_fea in class_feas:
        feas.remove(class_fea)
    print('make stat feas:', feas)

    def stat_fea(df):
        agg_dict = {
            'label': ['mean', 'min', 'max'],
            '湿球空气温度': ['mean', 'min', 'max'],
            '露点空气温度': ['mean', 'min', 'max'],
            '蒸气压': ['mean', 'min', 'max'],
            '相对湿度': ['mean'],
        }

        df_day = df.groupby(['站号', 'dayofyear']).agg(agg_dict)
        df_day.columns = ['_dayofyear_'.join(x) for x in df_day.columns]
        df_day = df_day.reset_index(drop=False)
        df = pd.merge(df, df_day, how='left', on=['站号', 'dayofyear'])

        df_hour = df.groupby(['站号', 'hour']).agg(agg_dict)
        df_hour.columns = ['_hour_'.join(x) for x in df_hour.columns]
        df_hour = df_hour.reset_index(drop=False)
        df = pd.merge(df, df_hour, how='left', on=['站号', 'hour'])

        df_month = df.groupby(['站号', 'day']).agg(agg_dict)
        df_month.columns = ['_day_'.join(x) for x in df_month.columns]
        df_month = df_month.reset_index(drop=False)
        df = pd.merge(df, df_month, how='left', on=['站号', 'day'])

        df_region = df.groupby(['站号', '地区']).agg(agg_dict)
        df_region.columns = ['_地区_'.join(x) for x in df_region.columns]
        df_region = df_region.reset_index(drop=False)
        df = pd.merge(df, df_region, how='left', on=['站号', '地区'])
        return df

    all_data = concat_train_and_test(train_X, test_X)
    # all_data = stat_fea(all_data)

    # # 对每个item都做滞后和滑动特征
    # for fea in feas:
    #     # shifts = [0, 6, 12, 24]
    #     shifts = [0]
    #     for shift in shifts:
    #         group_fea = ['站号', 'dayofyear', 'hour']
    #         all_data = makelag(all_data, all_data.groupby(group_fea), group_fea, fea, shift)

    # cv1:0.40634
    # train: before 2021.11
    # val 2021.11
    # test 2021.12

    # cv2: 0.08819	48fea
    # train: before 2020.12
    # val 2020.12
    # test 2021.12

    # cv3: 0.08821	66fea
    # train val : not 2021.12
    # test 2021.12

    # cv4:
    # train val : before 2020.12 include cvs
    # test 2021.12
    train_df = all_data[all_data['is_train'] == 1].drop(['is_train'], axis=1).reset_index(drop=True)
    test_df = all_data[all_data['is_train'] == 0].drop(['is_train', 'label'], axis=1).reset_index(drop=True)

    train_mask = (train_df['year'] < 2022)
    val_mask = (train_df['year'] == 2020) & (train_df['month'] == 12)

    X_train = train_df[train_mask].drop(['label'], axis=1).reset_index(drop=True)
    y_train = train_df[train_mask]['label']
    X_val = train_df[val_mask].drop(['label'], axis=1).reset_index(drop=True)
    y_val = train_df[val_mask]['label']

    X_test = test_df

    return X_train, y_train, X_val, y_val, X_test


def FE_scene_stat(train_X, train_y, test_X, ori_feas):
    # 场视角统计特征
    ## 活跃position数特征
    add = pd.DataFrame(data.groupby(["appID"]).positionID.nunique()).reset_index()
    add.columns = ["appID", "app_active_position"]
    data = data.merge(add, on=["appID"], how="left")




    train_X['label'] = train_y

    agg_dict = {'label': 'mean',
                '浏览量': 'mean',
                '抖音转化率': 'mean',
                '视频个数': 'mean',
                '直播个数': 'mean',
                '直播销量': 'mean',
                '视频销量': 'mean',
                '视频达人': 'mean',
                '直播达人': 'mean',
                }
    # log1p
    df = train_X.groupby(['商品id', 'week']).agg(agg_dict)
    # df.columns = ['_'.join(x) for x in df.columns]
    df = df.reset_index()

    # 对每个item都做滞后和滑动特征
    for fea in ['label',
                '浏览量',
                '抖音转化率',
                '视频个数',
                '直播个数',
                '直播销量',
                '视频销量',
                '视频达人',
                '直播达人'
                ]:
        df = makelag(df, df.groupby(['商品id']), fea, 1)
        df = makelag(df, df.groupby(['商品id']), fea, 2)

    df_X = df.drop('label', axis=1)
    le = LabelEncoder()
    df_X['商品id_int'] = le.fit_transform(df_X['商品id'].values)
    df_y = df[['商品id', 'week', 'label']]
    # cv
    # train:42-48 val:49
    # 42-49 50
    # 42-50 51
    # test 52
    df_X_train_1 = df_X.loc[df_X['week'] < 49].loc[df_X['week'] > 41].reset_index(drop=True)
    df_X_val_1 = df_X.loc[df_X['week'] == 49].reset_index(drop=True)
    df_X_train_2 = df_X.loc[df_X['week'] < 50].loc[df_X['week'] > 41].reset_index(drop=True)
    df_X_val_2 = df_X.loc[df_X['week'] == 50].reset_index(drop=True)
    df_X_train_3 = df_X.loc[df_X['week'] < 51].loc[df_X['week'] > 41].reset_index(drop=True)
    df_X_val_3 = df_X.loc[df_X['week'] == 51].reset_index(drop=True)

    df_y_train_1 = df_y.loc[df_y['week'] < 49].loc[df_y['week'] > 41]['label'].reset_index(drop=True)
    df_y_val_1 = df_y.loc[df_y['week'] == 49]['label'].reset_index(drop=True)
    df_y_train_2 = df_y.loc[df_y['week'] < 50].loc[df_y['week'] > 41]['label'].reset_index(drop=True)
    df_y_val_2 = df_y.loc[df_y['week'] == 50]['label'].reset_index(drop=True)
    df_y_train_3 = df_y.loc[df_y['week'] < 51].loc[df_y['week'] > 41]['label'].reset_index(drop=True)
    df_y_val_3 = df_y.loc[df_y['week'] == 51]['label'].reset_index(drop=True)

    X_train, y_train, X_val, y_val = [], [], [], []
    X_train.append(df_X_train_1)
    X_train.append(df_X_train_2)
    X_train.append(df_X_train_3)
    y_train.append(df_y_train_1)
    y_train.append(df_y_train_2)
    y_train.append(df_y_train_3)
    X_val.append(df_X_val_1)
    X_val.append(df_X_val_2)
    X_val.append(df_X_val_3)
    y_val.append(df_y_val_1)
    y_val.append(df_y_val_2)
    y_val.append(df_y_val_3)

    X_test = df_X.loc[df['week'] == 52]

    return X_train, y_train, X_val, y_val, X_test





# lagging + rolling
def makelag(data, group, group_fea, fea, shift):
    lags = [i + shift for i in range(1, 3)]
    # rollings = [i for i in range(2, 13)]
    for lag in lags:
        data[f'{fea}_lag_{lag}'] = group[fea].shift(lag)
    # for rolling in rollings:
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_min'] = group[fea].shift(shift).rolling(window=rolling).min()
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_max'] = group[fea].shift(shift).rolling(window=rolling).max()
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_median'] = group[fea].shift(shift).rolling(window=rolling).median()
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_std'] = group[fea].shift(shift).rolling(window=rolling).std()
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_mean'] = group[fea].shift(shift).rolling(window=rolling).mean()
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_skew'] = group[fea].shift(shift).rolling(window=rolling).skew()
    #     data[f'{group_fea}_{fea}_s_{shift}_roll_{rolling}_kurt'] = group[fea].shift(shift).rolling(window=rolling).kurt()
        # data[f'{fea}_s_{shift}_roll_{rolling}_cov'] = group[fea].shift(shift).rolling(window=rolling).cov()

    return data

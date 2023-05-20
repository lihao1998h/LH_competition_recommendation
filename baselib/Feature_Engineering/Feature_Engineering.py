import time

from baselib.Feature_Engineering.Feature_Engineering_base import base_feature, TS_feature
from baselib.Feature_Engineering.FE_ctr import *

# 3 特征工程 输入：train_X, train_y, test_X
# 输出：data_train, data_test
def feature_engineering(train_X, train_y, test_X, args):
    # 构造具有区分度的特征
    print('====================特征工程开始...================================')
    FE_time_start = time.time()
    ori_feas = train_X.columns.tolist()
    ori_feas.remove(args.time_series)

    # 基本特征
    all_X = concat_train_and_test(train_X, test_X)
    all_X = base_feature(all_X)
    all_X = TS_feature(all_X, 'args.time_series', level=2)


    train_X = all_X[all_X['is_train'] == 1].drop(['is_train'], axis=1)
    test_X = all_X[all_X['is_train'] == 0].drop(['is_train'], axis=1)

    # 单统计特征
    # 重点关注表间联立
    # 可考虑：各类总count、滑窗count（clickTime之前所有天的count）、label（点击率、转化率）的count/sum、商品平均转化时间
    X_train, y_train, X_test = FE_user_stat(train_X, train_y, test_X, ori_feas)  # 客视角统计特征
    X_train, y_train, X_test = FE_goods_stat(train_X, train_y, test_X, ori_feas)  # 货视角统计特征
    X_train, y_train, X_test = FE_scene_stat(train_X, train_y, test_X, ori_feas)  # 场视角统计特征

    # 交叉统计特征
    # for feat_1,feat_2 in[('positionID','advertiserID'),('userID','sitesetID'),('positionID','connectionType'),('userID','positionID'),
    #                  ('appPlatform','positionType'),('advertiserID','connectionType'),('positionID','appCategory'),('appID','age'),
    #                  ('userID', 'appID'),('userID','connectionType'),('appCategory','connectionType'),('appID','hour'),('hour','age')]:

    # for feat_1, feat_2, feat_3 in [('appID', 'connectionType', 'positionID'), ('appID', 'haveBaby', 'gender')]:




    # 特征选择
    # X_train = X_train.drop(['date', 'time_feat'], axis=1)
    # X_val = X_val.drop(['date', 'time_feat'], axis=1)
    # X_test = X_test.drop(['date', 'time_feat'], axis=1)

    importance_drop_feas = []
    print('importance drop feas num:', len(importance_drop_feas))
    X_train = X_train.drop(importance_drop_feas, axis=1)
    X_test = X_test.drop(importance_drop_feas, axis=1)


    FE_time_end = time.time()
    print('=================特征工程完成，耗时： ', FE_time_end - FE_time_start, '秒============================')
    return X_train, y_train, X_test




def change_index(df, level='day', mode=0, is_train=False):
    if level == 'month':
        if mode == 0:
            df['date'] = pd.to_datetime(df['date'])
            df['is_holiday'] = df['date'].map(lambda x: 1 if is_holiday(x) else 0)
            df['date'] = df['date'].astype('str')
            df['date'] = df['date'].map(lambda x: x[:-3]).values
            # 把日期粒度都设置为月
            # 生成label和一些统计量

            if is_train:
                agg_dict = {'label': 'sum',
                            'is_sale_day': {'sum', 'count', 'mean', 'std', 'skew'},
                            'is_holiday': {'sum', 'mean', 'std', 'skew'},
                            }
            else:
                agg_dict = {'is_sale_day': {'sum', 'count', 'mean', 'std', 'skew'},
                            'is_holiday': {'sum', 'mean', 'std', 'skew'},
                            }

            df_label = df.groupby(['product_id', 'date']).agg(agg_dict)
            df_kurt = df.groupby(['product_id', 'date'])[['is_sale_day', 'is_holiday']].apply(
                lambda x: x.kurt())

            df_label.columns = ['_'.join(x) for x in df_label.columns]
            df_label['is_holiday_kurt'] = df_kurt['is_holiday']
            df_label['is_sale_day_kurt'] = df_kurt['is_sale_day']

            df_label = df_label.reset_index()
            df = df.merge(df_label, on=['product_id', 'date'], how='left')

            if is_train:
                df = df.drop(['label', 'is_sale_day', 'is_holiday'], axis=1)
            else:
                df = df.drop(['is_sale_day', 'is_holiday'], axis=1)
        elif mode == 1:
            df['date'] = df['year'].astype(str) + '-' + df['month'].map(
                lambda x: '0' + str(x) if x <= 9 else str(x))

            df['Date'] = pd.to_datetime(df['date'])
            df['quarter'] = df.Date.dt.quarter
            df = df.drop('Date', axis=1)
            # del df['year']

            ############ 2. type-date ############
            # 该类型在该月的销量统计量
            df['type_order_sum'] = df.groupby(['type', 'date'])['order'].transform('sum').values
            df['type_order_mean'] = df.groupby(['type', 'date'])['order'].transform('mean').values
            df['type_order_std'] = df.groupby(['type', 'date'])['order'].transform('std').values
            df['type_order_median'] = df.groupby(['type', 'date'])['order'].transform('median').values
            df['type_order_max'] = df.groupby(['type', 'date'])['order'].transform('max').values
            df['type_order_min'] = df.groupby(['type', 'date'])['order'].transform('min').values
            # 销量在该类型中的比例
            df['order_ratio'] = df['order'].values / df['type_order_sum'].values

            df['stock_diff'] = df['end_stock'].values - df['start_stock'].values
            df['type_stock_diff_sum'] = df.groupby(['type', 'date'])['stock_diff'].transform('sum').values
            df['type_stock_diff_mean'] = df.groupby(['type', 'date'])['stock_diff'].transform(
                'mean').values
            df['type_stock_diff_std'] = df.groupby(['type', 'date'])['stock_diff'].transform('std').values
            df['type_stock_diff_median'] = df.groupby(['type', 'date'])['stock_diff'].transform(
                'median').values

            # stock_diff在该类型中的比例
            df['stock_diff_ratio'] = df['order'].values / df['type_order_sum'].values
    else:
        if mode == 0:
            df['date'] = pd.to_datetime(df['date'])
            if is_train:
                # 把日期粒度都设置为月
                # 生成label
                agg_dict = {'is_sale_day': {'sum', 'count'}}
                df_label = df.groupby(['product_id', 'date']).agg(agg_dict)
                df_label.columns = ['_'.join(x) for x in df_label.columns]
                df_label = df_label.reset_index()
                df = df.merge(df_label, on=['product_id', 'date'], how='left')
                # df = df.drop(['is_sale_day'], axis=1)
                df = df.drop(['label', 'is_sale_day'], axis=1)
            else:
                agg_dict = {'is_sale_day': {'sum', 'count'}}
                df_label = df.groupby(['product_id', 'date']).agg(agg_dict)
                df_label.columns = ['_'.join(x) for x in df_label.columns]
                df_label = df_label.reset_index()
                df = df.merge(df_label, on=['product_id', 'date'], how='left')
                df = df.drop('is_sale_day', axis=1)

    return df




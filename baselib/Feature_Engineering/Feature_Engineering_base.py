"""
<特征工程>
输入：all_data
输出：train_final.csv,test_final.csv或 train_for_tree.csv,test_for_tree.csv,train_for_lr.csv,...

>特征决定你的上限，模型只不过在无限逼近这个值罢了。
>有人总结 Kaggle 比赛是 “Feature 为主，调参和 Ensemble 为辅”
>之所以构造不同的数据集是因为，不同模型对数据集的要求不同

> 自动特征工程FeatureTools
"""

import pandas as pd
import numpy as np
import gc
import os
from baselib.utils import check_path
from datetime import date, timedelta, datetime
from statsmodels.tsa.seasonal import STL


def base_feature(all_X):
    box_fea = False  # 5.<特征编码、分箱>
    if box_fea:
        '''
        对于频数较少的那些分类变量可以归类到‘其他’pandas.DataFrame.replace后再进行编码。
        对于字符型的特征，要在编码后转换数据类型pandas.DataFrame.astype
        '''
        ## 5.1 编码
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

        le = LabelEncoder()
        data[feat_1 + '_' + feat_2] = le.fit_transform(data[feat_1].astype('str') + data[feat_2].astype('str'))
        # 对类别特征进行 OneEncoder
        # data = pd.get_dummies(data, columns=['model', 'brand', 'bodyType', 'fuelType',
        #                                      'gearbox', 'notRepairedDamage', 'power_bin'])

        ## 5.2 分箱
        # 等频分桶；
        # 等距分桶；
        # Best-KS 分桶（类似利用基尼指数进行二分类）；
        # 卡方分桶；
        # 连续特征离散化pandas.cut，然后通过pandas.Series.value_counts观察切分点，再将对应bins或者说区间的连续值通过pandas.DataFrame.replace或pandas.DataFrame.loc到离散点0,1,2,…后再进行编码。

        data.loc[data['age'] < 10, 'age'] = 0


    v_fea = False  # 暴力特征
    if v_fea:
        com_fea = [
            ['湿球空气温度', '露点空气温度', '蒸气压', '相对湿度'],
        ]
        for com_f in com_fea:
            for i in range(len(com_f)):
                all_X[f'{com_f[i]}_log1p'] = np.log1p(all_X[com_f[i]])
                all_X[f'{com_f[i]}_log'] = np.log(all_X[com_f[i]])
                for k in range(2, 6):
                    all_X[f'{com_f[i]}^{k}'] = all_X[com_f[i]] ** k
                for k in [-1, -2, 0.5, -0.5]:
                    all_X[f'{com_f[i]}^{k}'] = all_X[com_f[i]] ** k

        com_f = ['湿球空气温度', '露点空气温度', '蒸气压', '相对湿度',
                 '湿球空气温度^2', '湿球空气温度^-1',
                 '湿球空气温度_log1p',
                 '露点空气温度^2', '露点空气温度^-1',
                 '露点空气温度_log1p',
                 '蒸气压^2', '蒸气压^-1',
                 '蒸气压_log1p',
                 '相对湿度^2','相对湿度^-1',
                 '相对湿度_log1p']
        for i in range(len(com_f)):
            for j in range(i + 1, len(com_f)):
                all_X[f'{com_f[i]}+{com_f[j]}'] = all_X[com_f[i]] + all_X[com_f[j]]
                all_X[f'{com_f[i]}*{com_f[j]}'] = all_X[com_f[i]] * all_X[com_f[j]]

        for i in range(len(com_f)):
            for j in range(len(com_f)):
                if i != j:
                    all_X[f'{com_f[i]}-{com_f[j]}'] = all_X[com_f[i]] - all_X[com_f[j]]
                    all_X[f'{com_f[i]}/{com_f[j]}'] = all_X[com_f[i]] / all_X[com_f[j]]

    return all_X


def xy_feature(df_X, df_y):
    temp_df = df_X.copy()
    temp_df['y'] = df_y.values
    temp_df['y_max'] = temp_df.groupby(['product_id'])['y'].transform('max')
    temp_df['y_min'] = temp_df.groupby(['product_id'])['y'].transform('min')
    temp_df['y_std'] = temp_df.groupby(['product_id'])['y'].transform('std')
    temp_df['y_mean'] = temp_df.groupby(['product_id'])['y'].transform('mean')
    fea = temp_df[['product_id', 'y_max', 'y_min', 'y_std', 'y_mean']].drop_duplicates().reset_index(drop=True)

    df_X['y'] = df_y.values
    # lagging
    for i in range(1, 13):
        # todo 更少lag
        df_X[f'y_{i}_lag'] = df_X.groupby('product_id')['y'].shift(i).values

    # rolling
    for i in [3, 6, 12]:
        df_X[f'y_{i}_rol'] = df_X.groupby('product_id')['y'].rolling(i).mean().values

    # lag+rol
    for d_shift in [1, 3, 12]:
        for d_window in [3, 6, 12]:
            col_name = 'y_shift_' + str(d_shift) + '_rol_' + str(d_window)
            df_X[col_name] = df_X.groupby(['product_id'])['y'].transform(
                lambda x: x.shift(d_shift).rolling(d_window).mean()).values

    return fea




def TS_feature(df_X, time_feat, level=2):
    '''\
    通常需要训练集和测试集一起构造特征
    level:时间粒度，默认为日，在某些竞赛中为月 1:月 2：日 3：小时 4：分钟 5：秒
    '''

    ################### 1. date #############
    df_X['time_feat'] = pd.to_datetime(df_X[time_feat])

    def if_vocation(list_date):
        # 判断日期是否为节假日
        if_vocation = []
        for date in list_date.values.tolist():
            if is_workday(date):
                if_vocation.append(0)
            else:
                if_vocation.append(1)
        return if_vocation

    if level >= 1:
        # 月级特征
        # 1.1 时间特征
        df_X['month'] = df_X['time_feat'].dt.month
        df_X['year'] = df_X['time_feat'].dt.year

        # 1.2 布尔特征
        # 学校的寒暑假月
        # df_X['school_season'] = 0
        # df_X.loc[df_X.month.isin([1, 2, 7, 8]), 'school_season'] = 1

        # 春节月 # 是否年初 / 年末
        # df_X['spring_festival'] = 0
        # df_X.loc[df_X.month.isin([1, 2]), 'spring_festival'] = 1

        # 季节 345春 678夏 91011秋 1212冬
        df_X['spring_month'] = 0
        df_X['summer_month'] = 0
        df_X['autumn_month'] = 0
        df_X['winter_month'] = 0
        df_X.loc[df_X.month.isin([3, 4, 5]), 'spring_month'] = 1
        df_X.loc[df_X.month.isin([6, 7, 8]), 'summer_month'] = 1
        df_X.loc[df_X.month.isin([9, 10, 11]), 'autumn_month'] = 1
        df_X.loc[df_X.month.isin([1, 2, 12]), 'winter_month'] = 1

        # data['is_yanhai'] = list(map(lambda x: 1 if x in yanhaicity else 0, data['pro_id']))
    if level >= 2:
        # 日级特征
        # 1.1 时间特征
        df_X['day'] = df_X['time_feat'].dt.day
        # df_X['dofw'] = df_X['time_feat'].apply(lambda x: x.weekday())  # Weekly day
        df_X['dayofyear'] = df_X['time_feat'].dt.dayofyear
        # 1.2 布尔特征
        # 是否月初 / 月末
        # 是否周末
        # 是否节假日
        # 是否特殊日期

        # 1.3 时间差特征
        # 距离年初 / 年末的天数
        # 距离月初 / 月末的天数
        # 距离周末的天数
        # 距离节假日的天数
        # 距离特殊日期的天数

        # 根据日的季节特征
        Y = 2000
        # 0:winter, 1:spring, 2:summer, 3:autumn
        seasons = [(0, (date(Y, 1, 1), date(Y, 3, 20))),
                   (1, (date(Y, 3, 21), date(Y, 6, 20))),
                   (2, (date(Y, 6, 21), date(Y, 9, 22))),
                   (3, (date(Y, 9, 23), date(Y, 12, 20))),
                   (0, (date(Y, 12, 21), date(Y, 12, 31)))]

        def get_season(now):
            if isinstance(now, datetime):
                now = now.date()
            now = now.replace(year=Y)
            return next(season for season, (start, end) in seasons
                        if start <= now <= end)

        df_X['season'] = df_X['time_feat'].apply(get_season)
    if level >= 3:
        # 小时级特征
        # 1.1 时间特征
        df_X['hour'] = df_X[time_feat].dt.hour

        # 是否 8-11早上 / 12-15中午 /16-19下午 / else晚上

        df_X['morning'] = 0
        df_X['noon'] = 0
        df_X['afternoon'] = 0
        df_X['evening'] = 0
        df_X.loc[df_X.hour.isin([8, 9, 10, 11]), 'morning'] = 1
        df_X.loc[df_X.hour.isin([12, 13, 14, 15]), 'noon'] = 1
        df_X.loc[df_X.hour.isin([16, 17, 18, 19]), 'afternoon'] = 1
        df_X.loc[df_X.hour.isin([0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23]), 'evening'] = 1

    df_X = df_X.drop(['time_feat', time_feat], axis=1)

    return df_X



def trick_feature(all_X):
    # trick的思想：训练集上的情况也会在测试集上出现
    # 可考虑的trick：重复值特征
    dup_trick = False
    if dul_trick:
        subset = ['creativeID', 'positionID', 'adID', 'appID', 'userID']
        data['maybe'] = 0
        pos = data.duplicated(subset=subset, keep=False)  # 重复的都为True
        data.loc[pos, 'maybe'] = 1
        pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)  # 只有第一个重复的为True
        data.loc[pos, 'maybe'] = 2
        pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)  # 只有最后一个重复的为True
        data.loc[pos, 'maybe'] = 3

        #比较关键的一步，初赛刚发现trick时提升不多,经过onehot后提升近3个千分点
        features_trans = ['maybe']
        data = pd.get_dummies(data, columns=features_trans)
        data['maybe_0'] = data['maybe_0'].astype(np.int8)
        data['maybe_1'] = data['maybe_1'].astype(np.int8)
        data['maybe_2'] = data['maybe_2'].astype(np.int8)
        data['maybe_3'] = data['maybe_3'].astype(np.int8)

def get_timespan(df, dt, minus, periods, freq='D'):
    # print(dt)
    # out = df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq).strftime("%Y-%m-%d").tolist()]
    out = df[[dt + minus - i - 1 for i in range(periods)]]
    return out


def slide_window_feature(train_X_, train_y, date, is_train_val=True):
    # train_X['时间'] = train_X['时间'].map(str)
    X = {}
    train_X = train_X_.copy()
    features = train_X.columns.tolist()
    id = 'session_id'
    time = 'rank'
    features.remove(id)
    features.remove(time)

    # id 特征
    X[id] = train_X[id].drop_duplicates().values
    X['max_rank'] = train_X.groupby(['session_id'])['max'].max().values
    features.remove('max')

    # feature窗口统计量
    for fea in features:
        # 把日期作为列构造特征
        df_fea = train_X.set_index(
            [id, time])[fea].unstack(
            level=-1).fillna(0)
        df_fea.columns = df_fea.columns.get_level_values(0)

        for i in [3, 6, 12, 24, 36, 48]:
            # 过去窗口
            # 预测日期 的统计值
            tmp = get_timespan(df_fea, date, i, i)
            X['%s_diff_%s_mean' % (fea, i)] = tmp.diff(axis=1).mean(axis=1).values
            X['%s_mean_%s_decay' % (fea, i)] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['%s_mean_%s' % (fea, i)] = tmp.mean(axis=1).values
            X['%s_median_%s' % (fea, i)] = tmp.median(axis=1).values
            X['%s_min_%s' % (fea, i)] = tmp.min(axis=1).values
            X['%s_max_%s' % (fea, i)] = tmp.max(axis=1).values
            X['%s_std_%s' % (fea, i)] = tmp.std(axis=1).values
            X['%s_sum_%s' % (fea, i)] = tmp.sum(axis=1).values
            X['%s_skew_%s' % (fea, i)] = tmp.skew(axis=1).values
            X['%s_kurt_%s' % (fea, i)] = tmp.kurt(axis=1).values

            # 预测日期-k 的统计值
            # timedelta = timedelta(days=-7)
            for k in [3, 6, 9, 12]:
                timedalta = k
                tmp = get_timespan(df_fea, date + timedalta, i, i)
                # if fea == 'ambient':
                X['%s_diff_%s_mean_%s' % (fea, i, timedalta)] = tmp.diff(axis=1).mean(axis=1).values
                X['%s_std_%s_%s' % (fea, i, timedalta)] = tmp.std(axis=1).values
                X['%s_mean_%s_decay_%s' % (fea, i, timedalta)] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                    axis=1).values
                X['%s_mean_%s_%s' % (fea, i, timedalta)] = tmp.mean(axis=1).values
                X['%s_median_%s_%s' % (fea, i, timedalta)] = tmp.median(axis=1).values
                X['%s_min_%s_%s' % (fea, i, timedalta)] = tmp.min(axis=1).values
                X['%s_max_%s_%s' % (fea, i, timedalta)] = tmp.max(axis=1).values
                X['%s_sum_%s_2' % (fea, i)] = tmp.sum(axis=1).values
                X['%s_skew_%s' % (fea, i)] = tmp.skew(axis=1).values
                X['%s_kurt_%s' % (fea, i)] = tmp.kurt(axis=1).values

        # 预测日期前12天的特征值
        for i in range(1, 13):
            # 只要u_d和i_d
            X['%s_day_%s' % (fea, i)] = get_timespan(df_fea, date, i, 1).values.ravel()

        # for i in range(7):
        #     X['%s_mean_4_dow{}_2017'.format(fea, i)] = get_timespan(df_fea, date, 28 - i, 4, freq='7D').mean(axis=1).values
        # for i in range(7):
        #     X['%s_mean_20_dow{}_2017'.format(fea, i)] = get_timespan(df_fea, date, 140 - i, 20, freq='7D').mean(axis=1).values

        # 'popularity的涨幅占比'
        # data['huanbi_1_2popularity'] = (data['last_1_popularity'] - data['last_2_popularity']) / data[
        #     'last_2_popularity']
        # data['huanbi_2_3popularity'] = (data['last_2_popularity'] - data['last_3_popularity']) / data[
        #     'last_3_popularity']
        # data['huanbi_3_4popularity'] = (data['last_3_popularity'] - data['last_4_popularity']) / data[
        #     'last_4_popularity']
        # data['huanbi_4_5popularity'] = (data['last_4_popularity'] - data['last_5_popularity']) / data[
        #     'last_5_popularity']
        # data['huanbi_5_6popularity'] = (data['last_5_popularity'] - data['last_6_popularity']) / data[
        #     'last_6_popularity']

        # '同比一年前的增长'
        # data["increase16_4"]=(data["last_16_sale"] - data["last_4_sale"]) / data["last_16_sale"]

    # 'model 前两个月的销量差值'

    # id * feature特征统计值
    for fea in features:
        X[f'{fea}_mean'] = train_X.groupby('session_id')[fea].mean().values
        X[f'{fea}_median'] = train_X.groupby('session_id')[fea].median().values
        X[f'{fea}_min'] = train_X.groupby('session_id')[fea].min().values
        X[f'{fea}_max'] = train_X.groupby('session_id')[fea].max().values
        X[f'{fea}_std'] = train_X.groupby('session_id')[fea].std().values
        X[f'{fea}_skew'] = train_X.groupby('session_id')[fea].skew().values
        X[f'{fea}_kurt'] = train_X.groupby('session_id')[fea].apply(lambda x: x.kurt())

    # 预测日期前12天的label
    train_X['pm'] = train_y
    df_label = train_X.set_index(
        [id, time])['pm'].unstack(
        level=-1).fillna(0)
    df_label.columns = df_label.columns.get_level_values(0)
    for i in range(1, 13):
        X['%s_day_%s' % ('pm', i)] = get_timespan(df_label, date, i, 1).values.ravel()

    # id * label特征
    # label的统计特征
    for i in [3, 6, 12, 24]:
        # 过去窗口
        # 预测日期 的统计值
        tmp = get_timespan(df_label, date, i, i)
        X['pm_diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['pm_mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['pm_mean_%s' % i] = tmp.mean(axis=1).values
        X['pm_median_%s' % i] = tmp.median(axis=1).values
        X['pm_min_%s' % i] = tmp.min(axis=1).values
        X['pm_max_%s' % i] = tmp.max(axis=1).values
        X['pm_std_%s' % i] = tmp.std(axis=1).values
        X['pm_sum_%s' % i] = tmp.sum(axis=1).values
        X['%s_skew_%s' % (fea, i)] = tmp.skew(axis=1).values
        X['%s_kurt_%s' % (fea, i)] = tmp.kurt(axis=1).values

    # '环比'
    # 上升趋势还是下降趋势
    ## 一阶
    for i in [0, 12, 24, 36, 48, 60]:
        tmp = get_timespan(df_label, date + i, 13, 13)
        X[f'huanbi_1_{i}'] = tmp.iloc[:, -1].values / tmp.iloc[:, 0].values
    ## 二阶
    for i in [0, 12, 24, 36, 48]:
        X[f'huanbi_2_{i}'] = X[f'huanbi_1_{i}'] / X[f'huanbi_1_{i + 12}']
    ## 三阶
    for i in [0, 12, 24, 36]:
        X[f'huanbi_3_{i}'] = X[f'huanbi_2_{i}'] / X[f'huanbi_2_{i + 12}']
    # data['huanbi_3_4'] = data['last_3_sale'] / data['last_4_sale']
    # data['huanbi_4_5'] = data['last_4_sale'] / data['last_5_sale']
    # data['huanbi_5_6'] = data['last_5_sale'] / data['last_6_sale']

    X = pd.DataFrame(X)

    # 自定义特征
    X['curve_type'] = 0
    for id in range(295):
        X.loc[X['session_id'] == id, 'curve_type'] = dict[id][0]

    # 日期特征
    # X['date_month'] = date.month
    # X['date_day'] = date.day
    # X['date_dayofweek'] = date.weekday()
    # X['date_weekofyear'] = date.isocalendar()[1]

    # print('预测日期：', pd.date_range(date, periods=7).strftime("%Y-%m-%d").tolist())
    pred_date_list = [date - i - 1 for i in range(12)]
    print('预测日期：', pred_date_list)
    if is_train_val:
        idx = train_X[time].isin(pred_date_list)
        y = train_y[idx].values
        return X, y
    return X




# # 2 <加变量> <特征提取>（Feature Extraction）
# ## 2.1 <简化>
# train["SimplOverallQual"] = train.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
#                                                        4 : 2, 5 : 2, 6 : 2, # average
#                                                        7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
#                                                       })
# ## 2.2 <组合>（特征间的加减乘除）
# ### 2.2.1 <连续变量的组合>
# train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
# ### 2.2.2 <离散变量的组合>
#
# ## 2.3 <多项式>
# train["OverallQual-s2"] = train["OverallQual"] ** 2
# train["OverallQual-s3"] = train["OverallQual"] ** 3
# train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
# ## 2.4 <正则提取>
# # >正则提取
# # >str.split分割字符，再用pandas.DataFrame.add_prefix添加前缀成为新变量
#
# ## 2.5 <其它>
# # >如可以把一个变量是否为0作为特征
# def geo_feature():
#     # 地理位置相关特征
#     # 按经纬度进行网格划分，用word2vec及逆行训练得到编码向量
#     # 然后区域统计特征
#     pass
#
#
#
#
# # all_data.groupby.agg()
# data['count'] = 1
# tmp = data[data['goods_has_discount']==1].groupby(['customer_id'])['count'].agg({'goods_has_discount_counts':'count'}).reset_index()
# customer_all = customer_all.merge(tmp,on=['customer_id'],how='left')
#
# for col in ['aid','goods_id','account_id']:
#     result = logs.groupby([col,'day'], as_index=False)['isExp'].agg({
#         col+'_cnts'      : 'count',
#         col+'_sums'      : 'sum',
#         col+'_rate'      : 'mean'
#         })
#     result[col+'_negs'] = result[col+'_cnts'] - result[col+'_sums']
#     data = data.merge(result, how='left', on=[col,'day'])
#
# # 计算某品牌的销售统计量，同学们还可以计算其他特征的统计量
# # 这里要以 train 的数据计算统计量
# train_gb = train.groupby("brand")
# all_info = {}
# for kind, kind_data in train_gb:
#     info = {}
#     kind_data = kind_data[kind_data['price'] > 0]
#     info['brand_amount'] = len(kind_data)
#     info['brand_price_max'] = kind_data.price.max()
#     info['brand_price_median'] = kind_data.price.median()
#     info['brand_price_min'] = kind_data.price.min()
#     info['brand_price_sum'] = kind_data.price.sum()
#     info['brand_price_std'] = kind_data.price.std()
#     info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
#     all_info[kind] = info
# brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
# data = data.merge(brand_fe, how='left', on='brand')
#
# # >all_data.groupby.last/min/max....
# def create_time_feat(df_X, time_feat):
#

#
#
# def add_prefix(df_X, exclude_columns, prefix):
#     # 给特征增加前缀
#     if isinstance(exclude_columns, str):
#         exclude_columns = [exclude_columns]
#
#     column_names = [col for col in df_X.columns if col not in exclude_columns]
#     df_X.rename(columns=dict(zip(column_names, [prefix + name for name in column_names])), inplace=True)
#     return df_X
# # 2 <减变量><特征选择>（Feature Selection）
# ## 2.1 <降维算法>
#
# ## 2.2 <直接删除>
# # >利用好了，就可以删掉原始数据了
# data = data.drop(['creatDate', 'regDate', 'regionCode'], axis=1)
# ## 2.3 <过滤法>（filter）
# # >先选择后训练。按照评估准则对各个特征进行评分，然后按照筛选准则来选择特征。
# ## 2.4 <包裹法>（wrapper）
# # >一训练一筛选。根据学习器预测效果评分，每次选择若干特征，或者排除若干特征。一般性能比过滤法好，但计算开销较大。可以分为前向搜索、后向搜索及双向搜索。
# ## 2.5 <嵌入法>（embedded）
# # >先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小排序选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。常见的嵌入式方法有L1正则化。
# # >另外，该方法常用于处理稀疏表示和字典学习以及压缩感知。
# # >Lasso 回归和决策树可以完成嵌入式特征选择
# # >大部分情况下都是用嵌入式做特征筛选
#
# # 4 <数据标准化/归一化>
# '''
# 数据标准化/归一化的作用：
# 1. 在梯度下降中不同特征的更新速度变得一致，更容易找到最优解
#
#
# 应用场景：
# 通过梯度下降法求解的模型需要归一化：线性回归、逻辑回归、SVM、NN等
#
# 决策树不需要归一化，因为是否归一化不会改变信息增益
# '''
# # 4.1 <标准正态分布标准化>Z-Score Normalization
# Scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
# df_X[column]=StandardScaler().fit_transform(df_X[column][:,np.newaxis])
#
# X_scaled = preprocessing.scale(X)  # Scaler=scale(X, axis=0, with_mean=True, with_std=True, copy=True)
# '''
# z=(x-miu)/sigma
# '''
# ## 4.2 <Min-max归一化（0-1）>Min-Max Scaling
# Scaler=MinMaxScaler(feature_range=(0, 1), copy=True)
# Scaler=minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
# '''
# def scale_minmax(col):
#     return (col-col.min())/(col.max()-col.min())
# '''
# Scaler=MaxAbsScaler(copy=True)# [-1,1]
# Scaler=maxabs_scale(X, axis=0, copy=True)
# Scaler.fit(X_train)
# Scaler.transform(X_test)
# Scaler.fit_transform(X_train)
#
# #standardizing data实例
# saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
# low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
# high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)
#


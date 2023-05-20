"""
Preprocessing.py
<数据预处理>
输入：train,test
输出：all_data
"""

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np

# from sklearn.preprocessing import *


def concat_train_and_test(train, test):
    # 把训练集和测试集一起处理可以减少代码量
    all_data_temp = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])
    del train, test
    gc.collect()
    return all_data_temp.reset_index(drop=True)


# 2. <缺失值处理>
# 不处理（针对类似 XGBoost 等树模型）；
# 删除（缺失数据太多）；
# 插值补全，包括均值/中位数/众数/建模预测/多重插补/压缩感知补全/矩阵补全等；
# 分箱，缺失值一个箱；

def missing_data_process(df, method='fill'):
    '''
    method: 'fill1', 'fill2'，'drop1', 'drop2'
    '''

    if method == 'drop1':
        # 删除变量
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        df_processed = df.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    elif method == 'drop2':
        # 删除样本
        df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
        # .dropna()
        train_df = train_df[~train_df['order_detail_status'].isin([101])]  # 删除101订单状态订单
    elif method == 'fill1':
        # create an object and specify the required configuration
        # imputing numerical feature values
        imp_1 = SimpleImputer(missing_values=np.nan, strategy="mean")  # “mean”, “median”, “most_frequent”, and “constant”
        numerical_cols = list(data.select_dtypes(include='number').columns)
        data[numerical_cols] = imp_1.fit_transform(data[numerical_cols])
        # imputing categorical feature values
        imp_2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        categorical_cols = list(data.select_dtypes(include='object').columns)
        data[categorical_cols] = imp_2.fit_transform(data[categorical_cols])
    elif method == 'fill2':
        # 均值/众数 填充
        # 根据业务实际情况填充。
        # 统计量：众数、中位数、均值
        # 插值法填充：包括随机插值，多重差补法，热平台插补，拉格朗日插值，牛顿插值等
        # 模型填充：使用回归、贝叶斯、随机森林、决策树等模型对缺失数据进行预测。
        # 定值填充
        # df.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)
        ## value: .mean()/.median()/.mode()/0/-1/
        ## method: 'ffill'---前填充；'bfill'--后填充
        numeric_features = df.select_dtypes(exclude='category')  # [np.number]
        categorical_features = df.select_dtypes(include='category')  # [np.object]
        if categorical_features.shape[1] > 0:
            for cat_fea in categorical_features:
                df[cat_fea] = df[cat_fea].fillna(df[cat_fea].mode())
        if numeric_features.shape[1] > 0:
            for num_fea in numeric_features:
                df[num_fea] = df[num_fea].fillna(df[num_fea].mean())
    elif method == 'fill3':
        numeric_features = df.select_dtypes(exclude='category')  # [np.number]
        categorical_features = df.select_dtypes(include='category')  # [np.object]
        if categorical_features.shape[1] > 0:
            for cat_fea in categorical_features:
                df[cat_fea] = df[cat_fea].astype(str).fillna('0')
                df[cat_fea] = df[cat_fea].astype('category')
        if numeric_features.shape[1] > 0:
            for num_fea in numeric_features:
                df[num_fea] = df[num_fea].fillna(0)
        # np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        # int( 中位数/0.5 + 0.5 ) * 0.5
    elif method == 'front_fill':
        # 用前面的数据填充
        df.fillna(method='ffill')
    elif method == 'behind_fill':
        # 用后面的数据填充
        df.fillna(method='bfill')
    elif method == 'fill_0':
        df = df.fillna(0)

        # 还可以分组填充
        # 例如 features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
        # 保留缺失值，用'None'填充

        # 偏正态分布，使用均值代替，可以保持数据的均值；偏长尾分布，使用中值代替，避免受 outlier 的影响
    return df


# 3 <异常值和歧义值处理>
# 通过箱线图（或 3-Sigma）分析删除异常值；
## 删除
# talExposureLog = totalExposureLog.loc[(totalExposureLog.pctr<=1000)]

# df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
# df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
# 调整到一个范围内
# df['x'] = np.clip(df['x'].values, 0, 150) # 低于0的会变成0，高于150的会变成150
# 文本数据的清洗
# 在比赛当中，如果数据包含文本，往往需要进行大量的数据清洗工作。如去除HTML 标签，分词，拼写纠正, 同义词替换，去除停词，抽词干，数字和单位格式统一等
#
# for dataset in combine:
#     dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#
# pd.crosstab(train_df['Title'], train_df['Sex'])
#
# for dataset in combine:
#     dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
#  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#
#     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#
# train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

## 这里我包装了一个异常值处理的代码，可以随便调用。
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n


# train = outliers_proc(train, 'power', scale=3)


# 6  <不平衡数据处理>
# <过采样>
# from imblearn.over_sampling import RandomOverSampler
# X = df.iloc[:,:-1].values
# y = df['quality'].values
# ros = RandomOverSampler()
# X, y = ros.fit_sample(X, y)
#
# from imblearn.over_sampling import SMOTE
# smo = SMOTE(random_state=42)
# X_smo, y_smo = smo.fit_sample(X_train, y_train)


def skew_process(df, col):
    # 偏态分布处理
    # if skew

    # BOX-COX 转换（处理有偏分布）；
    # 长尾截断；
    # test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
    # normal = pd.DataFrame(train[quantitative])
    # normal = normal.apply(test_normality)
    # print(not normal.any())
    #
    df[col] = np.log1p(df[col])  # 对数转换

# 8 <时间特征处理>
# >.to_datetime处理后切片成年、周、日、小时等新特征。然后可以用.dt.days/years等来调用，如
# df['create_order_time'] = pd.to_datetime(df['create_order_time'])
# df['date'] = df['create_order_time'].dt.date
# df['day'] = df['create_order_time'].dt.day
# df['hour'] = df['create_order_time'].dt.hour

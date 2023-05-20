"""
数据探索性分析 EDA_Base.py
输入：数据文件
输出：df_train,df_test
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st






def one_key_EDA(df, output_name):
    # 一键生成EDA
    import pandas_profiling
    pfr = pandas_profiling.ProfileReport(df)
    pfr.to_file(output_name)


def easy_look(df):
    print("--------------------------Shape Of Data---------------------------------")
    print(df.shape)
    columns = df.columns.values.tolist()  # 获取所有的变量名
    print('变量列表：', columns)
    print('随机给几个样本')
    # df.head()  # 给前几个样本
    # df.tail()  # 给后几个样本
    print(df.sample(10))
    print("-------------------------------INFO--------------------------------------")
    print(df.info())
    numeric_features = df.select_dtypes(exclude='category')  # [np.number]
    categorical_features = df.select_dtypes(include=[np.object])  # [np.object]
    if numeric_features.shape[1] == 0:
        print('没有连续变量')
    else:
        print('连续变量的一些描述信息，如基本统计量、分布等。')
        print(df.describe())
    if categorical_features.shape[1] == 0:
        print('没有分类变量')
    else:
        print('所有变量的一些描述信息。')
        print(df.describe(include='all'))
    print('重复值统计（todo）')
    is_dup = False
    # idsUnique = len(set(train.Id)) # train['Id'].nunique()
    # idsTotal = train.shape[0]
    # idsDupli = idsTotal - idsUnique
    # print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
    # df.duplicated()

    print('--------------------缺失值统计--------------------------')
    is_missing = True
    ### 需要注意的是有些缺失值可能已经被处理过，可以用下条语句进行替换
    # Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
    #
    # credit.isnull().sum()/float(len(credit))
    #
    #
    # bar(todo)
    # missing = train.isnull().sum()
    # missing = missing[missing > 0]
    # missing.sort_values(inplace=True)
    # missing.plot.bar()
    #
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)
    # for col in df.columns:
    #     print(col, df[col].isnull().sum())
    #
    if df.isnull().sum().sum() == 0:
        is_missing = False
    return is_dup, is_missing


def single_variable_EDA(df, label, task_type):
    # 数字变量和字符变量分开处理
    y = df[label]
    numeric_features = df.select_dtypes(exclude='category')  # [np.number]
    categorical_features = df.select_dtypes(include='category')  # [np.object]
    if task_type == 'category':
        categorical_features.drop(label, axis=1)
        print('分类label分析')
        print(label + "的特征分布如下：")
        print("{}特征有个{}不同的值".format(label, df[label].nunique()))
        print(df[label].value_counts())
    elif task_type == 'regression':
        numeric_features.drop(label, axis=1)
        print('回归label分析')
        print(label + "的特征分布如下：")
        print("{}特征有个{}不同的值".format(label, df[label].nunique()))
        print(df[label].value_counts())
    elif task_type == 'time_series_pred':
        numeric_features = numeric_features.drop(label, axis=1)
        print('时间序列预测label分析')
        print('label: ', label + "的特征分布如下：")
        print("{}特征有个{}不同的值".format(label, df[label].nunique()))
        print('按值升序')
        print(df[label].value_counts(sort=False).sort_index())
        print('按频次降序')
        print(df[label].value_counts())
        print('按频率降序')
        print(df[label].value_counts(1))

    if task_type == 'time_series_pred':
        # 时间序列分析
        # gp = df.groupby(['cat_global', 'date_block_num']) \
        #     .agg(item_cnt_month=('item_cnt_day', 'sum'))
        # gp.reset_index(inplace=True)

        plt.rcParams['figure.facecolor'] = 'white'
        sns.lineplot(data=df, x='data_time', y='data_value', hue='meter')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    if categorical_features.shape[1] > 0:
        print('分类特征分析')
        for cat_fea in categorical_features:
            print(cat_fea + "的特征分布如下：")
            print("{}特征有个{}不同的值".format(cat_fea, df[cat_fea].nunique()))
            print(df[cat_fea].value_counts())
            # 箱图（分类变量）
            # var = 'region'
            # data = pd.concat([df['price'], df[var]], axis=1)
            # f, ax = plt.subplots(figsize=(8, 6))
            # fig = sns.boxplot(x=var, y="price", data=data)
            # # fig.axis(ymin=0, ymax=800000);

    if numeric_features.shape[1] > 0:
        print('数字特征分析')
        for num_fea in numeric_features:
            print(num_fea + "的特征分布如下：")
            print('异常值outlier(todo)')

            print('分布和偏态情况(todo)')
            print('{:15}'.format(num_fea),
                  '偏度Skewness: {:05.2f}'.format(df[num_fea].skew()),
                  '   ',
                  '散度Kurtosis: {:06.2f}'.format(df[num_fea].kurt())
                  )
            # if skew()
            skew = True
            plt.figure(1)
            plt.title(num_fea + 'kdeplot')
            sns.kdeplot(df[num_fea], shade=True)
            # plt.figure(2)
            # plt.title(num_fea + 'Johnson SU')
            # sns.distplot(df[num_fea], kde=False, fit=st.johnsonsu)
            plt.figure(3)
            plt.title(num_fea + 'Normal')
            sns.distplot(df[num_fea], kde=False, fit=st.norm)
            # plt.figure(4)
            # plt.title(num_fea + 'Log Normal')
            # sns.distplot(df[num_fea], kde=False, fit=st.lognorm)
            # plt.figure(5)
            # res = st.probplot(df[num_fea], plot=plt)


## 4.8 查看具体频数直方图
# plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
# plt.show()


# 5. 多变量探索
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


## 5.1 列表groupby玩法

def groupby_cnt_ratio(df, col):
    # 单变量聚合
    if isinstance(col, str):
        col = [col]
    key = ['is_train'] + col

    # groupby function
    cnt_stat = df.groupby(key).size().to_frame('count')
    ratio_stat = (cnt_stat / cnt_stat.groupby(['is_train']).sum()).rename(
        columns={'count': 'count_ratio'})
    return pd.merge(cnt_stat, ratio_stat, on=key, how='outer').sort_values(by=['count'], ascending=False)


# df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# #count/sum/mean/median/std/var/min/max/first/last
# def groupby_cnt_ratio(df, col=[]):
#     if isinstance(col, str):
#         col = [col]
#     key = ['is_train', 'buyer_country_id'] + col
#
#     # groupby function
#     cnt_stat = df.groupby(key).size().to_frame('count')
#     ratio_stat = (cnt_stat / cnt_stat.groupby(['is_train', 'buyer_country_id']).sum()).rename(columns={'count':'count_ratio'})
#     return pd.merge(cnt_stat, ratio_stat, on=key, how='outer').sort_values(by=['count'], ascending=False)
## 5.2 相关性分析
# price_numeric = Train_data[numeric_features]
# correlation = price_numeric.corr()
# print(correlation['price'].sort_values(ascending = False),'\n')




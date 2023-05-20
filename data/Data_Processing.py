import time
from baselib.Data_Processing.Preprocessing_base import concat_train_and_test, missing_data_process
import numpy as np
from sklearn.preprocessing import StandardScaler


# 2 数据预处理 输入：df_train,df_test 输出：df_train_X, df_train_y, df_test_X
def data_processing(df_train, df_test, args):
    feature_test = df_test.columns
    print('将训练集和测试集合并处理')
    all_data = concat_train_and_test(df_train, df_test)

    print('====================数据预处理开始...================================')
    DP_time_start = time.time()

    mean, std = all_data[args.label].mean(), all_data[args.label].std()
    if args.log_label:
        # all_data[args.label] = np.log1p(all_data[args.label])
        all_data[args.label] = all_data[args.label].apply(lambda x: (x - mean) / std)
        print('logged label')

    if args.normalize_feature:
        feas = all_data.columns.tolist()
        log_feas = ['降水量', '相对湿度', '平均海平面压力', '平均每小时风速']
        for fea in log_feas:
            all_data[fea] = np.log1p(all_data[fea])
            # df_train[fea] = df_train[fea].replace([np.inf, -np.inf], np.nan)  # inf值处理
        print('log feas: ', log_feas)

        scale_feas = ['湿球空气温度', '露点空气温度', '蒸气压']
        ss_fea = StandardScaler()
        all_data[scale_feas] = ss_fea.fit_transform(all_data[scale_feas])
        print('scale feas: ', scale_feas)

        other_feas = ['主要每小时风向']
        for fea in other_feas:
            all_data[fea] = all_data[fea] / 360

    print('训练集测试集分布一致化预处理')  # todo
    if args.is_same_distribution:
        pass

    if args.is_dup:
        print('重复值删除，原始长度为')

        print('all_data has {} rows and {} columns'.format(all_data.shape[0], all_data.shape[1]))
        all_data = all_data.drop_duplicates()

        print('去重后 all_data has {} rows and {} columns'.format(all_data.shape[0], all_data.shape[1]))

    if args.is_missing:
        print('缺失值处理')
        print('原始缺失值:', all_data.isnull().sum().max())
        all_data_processed = missing_data_process(all_data, method='fill_0')
        print('final check(must be 0):', all_data_processed.isnull().sum().max())
        # print(len(all_data_processed['meter'].unique()))
    else:
        all_data_processed = all_data

    # abnormal_value = True
    # if abnormal_value:
    #     print('异常值处理')
    #     all_data_processed.loc[all_data_processed['data_value'] < 0, 'data_value'] = 0

    # 通过Z-Score方法判断异常值
    # df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
    # cols = df.columns  # 获得列表框的列名
    # for col in cols:
    #     df_col = df[col]  # 得到每一列的值
    #     z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每一列的Z-score得分
    #     df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分是否大于2.2，如果是则是True，否则为False
    # df_zscor

    # print('偏态分布处理')  # todo
    # np.log1p
    DP_time_end = time.time()
    print('=================数据预处理完成，耗时： ', DP_time_end - DP_time_start, '秒============================')

    train_data = all_data_processed[all_data_processed['is_train'] == 1]
    print('预处理后 train_data has {} rows and {} columns'.format(all_data.shape[0], all_data.shape[1]))

    test_data = all_data_processed[all_data_processed['is_train'] == 0]
    print('预处理后 test_data has {} rows and {} columns'.format(all_data.shape[0], all_data.shape[1]))
    train_X = train_data.drop([args.label, 'is_train'], axis=1)
    train_y = train_data[args.label]
    test_X = test_data[feature_test]

    return train_X, train_y, test_X, (mean, std)

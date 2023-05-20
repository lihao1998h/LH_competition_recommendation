from setting import args  # 参数设定

# 特征工程
from baselib.Feature_Engineering.Feature_Engineering import feature_engineering
# 模型训练验证
from train import train_val
from plot import plot_fig
from matplotlib import pyplot as plt
# import torch
# import torch.nn.functional as F
from baselib.utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from data.get_dataset import get_dataset
from model.get_model import get_model
from test import test_and_submit


today = datetime.date.today()

# df_sub_sample = pd.read_csv('./data/地区温度预测挑战赛公开数据' + '/sample_submit.csv', header=0)

task = 'task1'
pd.read_csv(os.path.join(args.data_root, f'sessions_test_{task}.csv'))

args.locales = ['UK', 'DE', 'JP', 'IT', 'FR', 'ES']

def main(args):
    train_loader, val_loader, test_loader = get_dataset(args)  # 包括数据读取和预处理，需要根据任务修改
    model = get_model(args)

    best_val_loss = train_val(model, train_loader, val_loader, args)
    # todo:利用pre的顺序

    test_and_submit(model, test_loader, args)



    # prepare_data = True  # 如果训练数据已被保存，则改为False
    # if prepare_data:
    #     rule = False  # rule
    #     if rule:
    #         pass
    #
    #     lgb = True
    #
    #     X_train, y_train, X_test = feature_engineering(train_X, train_y, test_X, args)
    #
    #     # 划分train-val：要么根据日期，要么直接split
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, X_train['label'], test_size=0.3,
    #                                                       random_state=seed)
    #
    #
    #     X_train.to_csv('./data/X_train.csv', index=None)
    #     y_train.to_csv('./data/y_train.csv', index=None)
    #     X_val.to_csv('./data/X_val.csv', index=None)
    #     y_val.to_csv('./data/y_val.csv', index=None)
    #     X_test.to_csv('./data/X_test.csv', index=None)
    #     data_train = []
    #     data_train.append(X_train)
    #     data_train.append(y_train)
    #     data_train.append(X_val)
    #     data_train.append(y_val)
    #     print('data saved')
    # else:
    #     # 直接读取数据
    #     X_train = pd.read_csv('./data/X_train.csv')
    #     y_train = pd.read_csv('./data/y_train.csv')
    #     X_val = pd.read_csv('./data/X_val.csv')
    #     y_val = pd.read_csv('./data/y_val.csv')
    #     data_test = pd.read_csv('./data/data_test.csv')
    #     data_train = []
    #     data_train.append(X_train)
    #     data_train.append(y_train)
    #     data_train.append(X_val)
    #     data_train.append(y_val)
    #     print('data loaded')
    # fea_num = data_train[0].shape[1]
    # val_loss, y_pred, val_pred = train_val(data_train, data_test, args)
    # if args.log_label:
    #     y_pred = y_pred * std + mean
    #     val_pred = val_pred * std + mean
    #
    #     val_loss = mean_absolute_error(val_pred, y_val)
    #     print("val mae loss: {:<8.8f}".format(val_loss))
    #
    #     # y_pred = np.expm1(y_pred)
    # # submission
    #
    # # for i in range(1, 13):
    # #     cols = y_pred.columns.tolist()
    # #     aaa = df_submit['rank'] == i
    # #     df_submit[df_submit['rank'] == i]['pm'] = y_pred[cols[i]]
    #
    # df_sub_sample[args.label] = y_pred
    # # df_submit['pm'] = df_submit['pm'].map(lambda x: 0 if x < 0 else int(x))
    # df_sub_sample.to_csv(f'output/lgb_notrec_{today}_fea_{fea_num}_val_{val_loss:<8.8f}.csv', index=None)
    # # .map(round)
    # pred_lgb = df_sub_sample
    #
    # pred_lgb.to_csv(f'output/pred_lgb.csv', index=None)


    # df_train = pd.read_csv('E:/data/地区温度预测挑战赛公开数据/train.csv', index_col=None)
    # df_val_pred = df_train.loc[df_train['年'] == 2020].loc[df_train['月'] == 12].reset_index(drop=True)
    # df_val_pred[args.label] = val_pred
    # df_val_pred = df_val_pred[['站号', '日', '气温']]
    # df_val_pred.to_csv(f'output/pred_val.csv', index=None)


    # 5 增强方法
    # 5.1 时序stacking
    # train_data1, train_data2, test_data
    ## step1
    # model_meta1 = train_model(train_data1)
    # model_meta2 = train_model(train_data2)
    # meta_feature1 = model_meta1.predict(train_data2)
    # meta_feature2 = model_meta2.predict(test_data)
    # ## step2
    # model = train_model([train_data2, meta_feature1])
    # pred = model.predict([test_data, meta_feature2])



def dateRange(start, end, step=1, format="%Y/%m/%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]


if __name__ == '__main__':
    main(args)

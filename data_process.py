"""
讯飞-地区温度预测挑战赛

feature :   降水量,湿球空气温度,露点空气温度,蒸气压,相对湿度,
            平均海平面压力,平均每小时风速,主要每小时风向,
            地区,站号,
            年,月,日

plot    :   按月出图

pred    :   23站点 * 31天 * 24小时

0905    :   arima 月数据 m=12

"""

"""
time series predict
get_feature():

相同类 聚合 训练
递归 补数据 训练
堆成一行 训练
1-12 12个模型

"""

import pandas as pd


class time_series_dataframe(object):
    def __init__(self,
                 train_data_path: str,
                 test_data_path: str,
                 submit_data_path: str,
                 label: str):
        self.tra_path = train_data_path
        self.te_path = test_data_path
        self.sub_path = submit_data_path

        self.df_train = pd.read_csv(self.tra_path, index_col=None)
        self.df_test = pd.read_csv(self.te_path, index_col=None)
        self.df_submit = pd.read_csv(self.sub_path, index_col=None)
        self.df = self.__aggregate__()
        self.df_use = None

        self.label = label
        self.feature = [c for c in self.df.columns if c not in [label]]

    def __aggregate__(self):
        """
            aggregate df_train df_test
        """
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        df_train['is_train'] = 1
        df_test['is_train'] = 0
        df = pd.concat([df_train, df_test], axis=0)
        # df.sort_values(by=sort_list, inplace=True)
        # df.reset_index(inplace=True, drop=True)
        return df

    def _get_id_df_class(self, id_col_name, id):
        return time_series_id_dataframe(self.tra_path,
                                        self.te_path,
                                        self.sub_path,
                                        label=self.label,
                                        id_col_name=id_col_name,
                                        id=id)


class time_series_id_dataframe(time_series_dataframe):
    def __init__(self, train_data_path: str, test_data_path: str, submit_data_path: str, label: str, id_col_name, id):
        super().__init__(train_data_path, test_data_path, submit_data_path, label)
        self.id_col_name = id_col_name
        self.id = id
        self.df_id = self._get_id_df()

    def _get_id_df(self):
        df = self.df
        return df.loc[df[self.id_col_name] == self.id]

    def _construct_feature(self,
                           shift=1,
                           start=0,
                           lagging_num=15,
                           rolling_num=15,
                           pct_num=15,
                           method='label',
                           data_to_construct='id'
                           ):
        """
        method choose 'id' means to construct feature base on id
        method choose 'all' means to construct feature base on all feature
        """
        if method == 'all':
            self.feature.append(self.label)
        elif method == 'label':
            self.feature = [self.label]
        for feature_name in self.feature:
            self.df_use = self._make_lag(feature_name,
                                         shift=shift,
                                         start=start,
                                         lagging_num=lagging_num,
                                         data_to_construct=data_to_construct
                                         )
            self.df_use = self._make_rolling(feature_name,
                                             shift=shift,
                                             start=start,
                                             rolling_num=rolling_num,
                                             data_to_construct=data_to_construct
                                             )
            self.df_use = self._make_pct(feature_name,
                                         shift=shift,
                                         start=start,
                                         pct_num=pct_num,
                                         data_to_construct=data_to_construct
                                         )
            self.df_use = self._make_normalize(feature_name,
                                               data_to_construct=data_to_construct
                                               )
        self.feature = [c for c in self.df_use.columns if c not in [self.label]]

    def _make_normalize(self,
                        feature_name,
                        data_to_construct='id'):
        if data_to_construct == 'id':
            data = self.df_id
        elif data_to_construct == 'all':
            data = self.df
        else:
            raise ValueError("error-parameter:data_to_construct")
        values = data[feature_name]

        std = values.transform('std')
        mean = values.transform('mean')

        data[feature_name] = (data[feature_name] - mean) / std

        return data

    def _make_lag(self,
                  feature_name,
                  shift=1,
                  start=0,
                  lagging_num=15,
                  data_to_construct='id'):
        if data_to_construct == 'id':
            data = self.df_id
        elif data_to_construct == 'all':
            data = self.df
        else:
            raise ValueError("error-parameter:data_to_construct")
        values = data[feature_name]
        shift = shift + start
        lags = [i + shift for i in range(lagging_num)]
        for lag in lags:
            data[f'lag_{feature_name}_{lag}_s_{shift}'] = values.shift(lag)

        return data

    def _make_rolling(self,
                      feature_name,
                      shift=1,
                      start=0,
                      rolling_num=15,
                      data_to_construct='id'):
        if data_to_construct == 'id':
            data = self.df_id
        elif data_to_construct == 'all':
            data = self.df
        else:
            raise ValueError("error-parameter:data_to_construct")
        values = data[feature_name]
        shift = shift + start
        rollings = [i for i in range(2, rolling_num)]
        for rolling in rollings:
            data[f'{feature_name}_s_{shift}_roll_{rolling}_min'] = values.shift(shift).rolling(window=rolling).min()
            data[f'{feature_name}_s_{shift}_roll_{rolling}_max'] = values.shift(shift).rolling(window=rolling).max()
            data[f'{feature_name}_s_{shift}_roll_{rolling}_median'] = values.shift(shift).rolling(
                window=rolling).median()
            data[f'{feature_name}_s_{shift}_roll_{rolling}_std'] = values.shift(shift).rolling(window=rolling).std()
            data[f'{feature_name}_s_{shift}_roll_{rolling}_mean'] = values.shift(shift).rolling(window=rolling).mean()

            data[f'{feature_name}_s_{shift}_roll_{rolling}_skew'] = values.shift(shift).rolling(window=rolling).skew()
            data[f'{feature_name}_s_{shift}_roll_{rolling}_kurt'] = values.shift(shift).rolling(window=rolling).kurt()
            data[f'{feature_name}_s_{shift}_roll_{rolling}_cov'] = values.shift(shift).rolling(window=rolling).cov()

        return data

    def _make_pct(self,
                  feature_name,
                  shift=1,
                  start=0,
                  pct_num=15,
                  data_to_construct='id'):
        if data_to_construct == 'id':
            data = self.df_id
        elif data_to_construct == 'all':
            data = self.df
        else:
            raise ValueError("error-parameter:data_to_construct")

        values = data[feature_name]
        shift = shift + start
        rollings = [i for i in range(1, pct_num)]
        for rolling in rollings:
            data[f'{feature_name}_roll_{rolling}_pct'] = values.pct_change(periods=rolling)
            df_temp = data[f'{feature_name}_roll_{rolling}_pct']
            data[f'{feature_name}_roll_{rolling}__pct_min'] = df_temp.rolling(window=rolling).min()
            data[f'{feature_name}_roll_{rolling}__pct_max'] = df_temp.rolling(window=rolling).max()
            data[f'{feature_name}_roll_{rolling}__pct_median'] = df_temp.rolling(window=rolling).median()
            data[f'{feature_name}_roll_{rolling}__pct_std'] = df_temp.rolling(window=rolling).std()
            data[f'{feature_name}_roll_{rolling}__pct_mean'] = df_temp.rolling(window=rolling).mean()

            data[f'{feature_name}_s_{shift}_roll_{rolling}__pct_skew'] = df_temp.rolling(window=rolling).skew()
            data[f'{feature_name}_s_{shift}_roll_{rolling}__pct_kurt'] = df_temp.rolling(window=rolling).kurt()
            data[f'{feature_name}_s_{shift}_roll_{rolling}__pct_cov'] = df_temp.rolling(window=rolling).cov()

        return data



class time_series_lihao(time_series_dataframe):
    def __init__(self, train_data_path: str, test_data_path: str, submit_data_path: str, label: str, args=None):
        super().__init__(train_data_path, test_data_path, submit_data_path, label)
        self.args=args

    def _all_id_con_fea(self):
        re = self.xxx.make_all_per()
        return re
    # def make_all_per(self, num, ):
    #     ...""""""


class time_series_model(object):
    def __init__(self,
                 train_data: time_series_dataframe,
                 model_name):
        self.tra_data = train_data
        self.model_name = model_name  # [prophet, arima]

    def _prophet(self):
        df_model_1 = self.tra_data[[args.time_series, args.label]]
        df_model_1.rename(columns={args.time_series: 'ds'}, inplace=True)
        df_model_1.rename(columns={args.label: 'y'}, inplace=True)
        model = Prophet()
        model.fit(df_model_1)
        future = model.make_future_dataframe(periods=args.pred_time, freq='h')
        pred = model.predict(future)
        pred = pred.iloc[-args.pred_time:]['yhat'].values
        df_sub_sample.loc[df_sub_sample[args.class_name] == id, args.label] = pred
        print(f'prophet finished {id}')

        print('prophet finished')
        pred_prophet = df_sub_sample
        pred_prophet.to_csv(f'output/pred_prophet.csv', index=None)


if __name__ == '__main__':
    df = time_series_dataframe(train_data_path='./地区温度预测挑战赛公开数据/train.csv',
                               test_data_path='./地区温度预测挑战赛公开数据/test.csv',
                               submit_data_path='./地区温度预测挑战赛公开数据/sample_submit.csv',
                               label='气温')


    for index, id in df.df['站号'].iteritems():
        df_id = df._get_id_df_class(id_col_name='站号', id=id)
        df_id._construct_feature()
        df_use = df_id.df_use

    # df_use.concat([])


    # df.df_use  = con~
    # df.df_use._all_id_con_fea

    print(df)

'''
def _get_train_test_aggregate(df_train_read, df_test_read, sort_list: list):
    """
    aggregate
    """
    df_train = df_train_read.copy()
    df_test = df_test_read.copy()
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df = pd.concat([df_train, df_test], axis=0)
    df.sort_values(by=sort_list, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


# lagging + rolling
def _make_lag(data, values, shift, name, max_):
    lags = [i + shift for i in range(15)]
    for lag in lags:
        data[f'lag_{name}_{lag}'] = values.shift(lag)

    return data


def _make_rolling(data, values, shift, name):
    rollings = [i for i in range(2, 15)]
    for rolling in rollings:
        data[f'{name}_s_{shift}_roll_{rolling}_min'] = values.shift(shift).rolling(window=rolling).min()
        data[f'{name}_s_{shift}_roll_{rolling}_max'] = values.shift(shift).rolling(window=rolling).max()
        data[f'{name}_s_{shift}_roll_{rolling}_median'] = values.shift(shift).rolling(window=rolling).median()
        data[f'{name}_s_{shift}_roll_{rolling}_std'] = values.shift(shift).rolling(window=rolling).std()
        data[f'{name}_s_{shift}_roll_{rolling}_mean'] = values.shift(shift).rolling(window=rolling).mean()

        data[f's_{shift}_roll_{rolling}_skew'] = values.shift(shift).rolling(window=rolling).skew()
        data[f's_{shift}_roll_{rolling}_kurt'] = values.shift(shift).rolling(window=rolling).kurt()
        data[f's_{shift}_roll_{rolling}_cov'] = values.shift(shift).rolling(window=rolling).cov()

    return data


def _make_normalize(data, values, shift, name):
    std = data.groupby(['session_id'])[name].transform('std')
    mean = data.groupby(['session_id'])[name].transform('mean')

    data[name] = (data[name] - mean) / std

    return data
'''
'''
def makepct(data, values, shift, name):
    rollings = [i for i in range(1, 13)]
    for rolling in rollings:
        data[f'{name}_roll_{rolling}_pct'] = values.pct_change(periods=rolling)
        data[f'{name}_roll_{rolling}_min'] = data[f'{name}_roll_{rolling}_pct'].rolling(window=rolling).min()
        data[f'{name}_roll_{rolling}_max'] = data[f'{name}_roll_{rolling}_pct'].rolling(window=rolling).max()
        data[f'{name}_roll_{rolling}_median'] = data[f'{name}_roll_{rolling}_pct'].rolling(window=rolling).median()
        data[f'{name}_roll_{rolling}_std'] = data[f'{name}_roll_{rolling}_pct'].rolling(window=rolling).std()
        data[f'{name}_roll_{rolling}_mean'] = data[f'{name}_roll_{rolling}_pct'].rolling(window=rolling).mean()

        data[f's_{shift}_roll_{rolling}_skew'] = values.shift(shift).rolling(window=rolling).skew()
        data[f's_{shift}_roll_{rolling}_kurt'] = values.shift(shift).rolling(window=rolling).kurt()
        data[f's_{shift}_roll_{rolling}_cov'] = values.shift(shift).rolling(window=rolling).cov()

    return data


def get_df():
    # read data
    df_train_read = pd.read_csv('./地区温度预测挑战赛公开数据/train.csv', index_col=None)
    df_test_read = pd.read_csv('./地区温度预测挑战赛公开数据/test.csv', index_col=None)

    # construct feature
    df = _get_train_test_aggregate(df_train_read, df_test_read, sort_list=['站号', '年', '月', '日'])

    return df


def train(df):
    set_seed(SEED)

    model_name = 'arima'  # 影响输出时名称
    df_base = pd.read_csv('arima-all-base.csv', index_col=None)  # 最高分基线
    df_mixed = pd.DataFrame()

    df_final = pd.read_csv('lgb_final.csv', index_col=None)

    for id in df['session_id'].unique():
        if True:
            # 按-自选-模型输出
            pred = model_arima(df, id)
            new_base = df_base.loc[df_base['session_id'] == id].copy()
            new_base['pm-' + model_name] = np.array(pred)[::-1]
            df_mixed = new_base if df_mixed.empty else pd.concat([df_mixed, new_base])
        else:
            # retain
            new_base = df_base.loc[df_base['session_id'] == id].copy()
            new_base['pm-' + model_name] = new_base['pm']
            del new_base['pm']
            df_mixed = new_base if df_mixed.empty else pd.concat([df_mixed, new_base])

    df_sub = df_mixed.loc[:, ['session_id', 'rank', 'pm-' + model_name]]
    df_sub.rename(columns={'pm-' + model_name: 'pm'}, inplace=True)
    time = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H-%M')
    df_sub.to_csv(model_name + '-' + str(time) + '.csv', index=None)


def model_prophet(df, id, cut_off=True):
    df_model_1 = df.loc[df['rank'] > 12]
    if cut_off and dict[id][2] != -1:
        df_model_1 = df_model_1.loc[df_model_1['rank'] <= dict[id][2]]
    df['date'] = df['seq_id'].map(lambda x: datetime.datetime.now() + datetime.timedelta(days=x))
    df_model_1 = df_model_1.loc[df_model_1['session_id'] == id][['date', 'pm']]
    df_model_1.rename(columns={'date': 'ds'}, inplace=True)
    df_model_1.rename(columns={'pm': 'y'}, inplace=True)
    model = Prophet()
    model.fit(df_model_1)
    future = model.make_future_dataframe(periods=12, freq='d')
    pred = model.predict(future)
    pred = pred.iloc[-12:].loc[:, 'yhat']
    return pred


def model_arima(df, id, cut_off=True, recursion=False):
    """
    cut_off : 是否按dict截断
    recursion : 是否递归预测
    """

    df = df.loc[df['session_id'] == id]
    # df = _get_lgbm_features(df)
    features = [c for c in df.columns if c not in ['pm', 'is_train', 'max',
                                                   'u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth',
                                                   'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient',
                                                   'torque'
                                                   ]]
    label = 'pm'

    df_train = df.loc[df['rank'] > 12]
    df_test = df.loc[df['rank'] <= 12]
    if cut_off and dict[id][2] != -1:
        df_train = df_train.loc[df_train['rank'] <= dict[id][2]]

    df_tr_label = df_train[label]
    df_tr_feature = df_train[features].fillna(method="bfill").fillna(0)
    df_te_feature = df_test[features].fillna(method="ffill")

    if recursion:
        df_model_1 = df.loc[df['rank'] > 12]
        df_model_1 = df_model_1.loc[df_model_1['session_id'] == id]['pm']
        pred = []
        for rank in range(1, 13):
            model = auto_arima(df_model_1, start_P=0, start_q=0)
            model.fit(df_model_1)
            pred_next = model.predict(12)
            pred.append(pred_next[0])
            df_model_1 = df_model_1.append(pd.Series(pred_next[0]))
        return pred

    model = auto_arima(y=df_tr_label, start_p=0, start_q=0, max_p=6, max_q=6, max_d=2,
                       with_intercept=True,
                       random_state=42, out_of_sample_size=12, scoring='mae',

                       seasonal=True, test='adf',
                       error_action='ignore',
                       information_criterion='aic',
                       njob=-1, suppress_warnings=True)

    model.fit(df_tr_label)
    pred = model.predict(12)

    return pred


def model_lgbm(df, id, cut_off=True, recursion=False, version='v1'):
    lgb_params = {
        'boosting_type': 'gbdt',  # options: gbdt, rf, dart, goss
        'num_leaves': 2 ** 10 - 1,  # max number of leaves in one tree
        'num_boost_round': 20000,
        'min_data_in_leaf': 50,  # 叶节点最小值，用于缓解过拟合
        'objective': 'mae',  # todo 试试poisson和tweedie
        # options: regression/mse/rmse, regression_l1/mae, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg
        # 'tweedie_variance_power': 1.1,
        'max_depth': 10,
        'learning_rate': 0.01,
        "feature_fraction": 0.8,  # 默认值为1；指定每次迭代所需要的特征部分
        "bagging_freq": 1,  # k 意味着每 k 次迭代执行bagging,0表示不bagging
        "bagging_fraction": 0.8,  # 默认值为1；指定每次迭代所需要的数据部分，并且它通常是被用来提升训练速度和避免过拟合的。
        "bagging_seed": SEED,
        "metric": 'mae',  # 常用：mse,mae,rmse,mape  https://zhuanlan.zhihu.com/p/91511706
        "lambda_l1": 0.1,
        "verbosity": -1,  # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug 默认1
        'early_stopping_rounds': 100,
        # 'colsample_bytree': 0.95,
        'seed': SEED
    }
    df = df.loc[df['session_id'] == id]

    if cut_off and dict[id][2] != -1:
        df = df.loc[df['rank'] <= dict[id][2]]

    if recursion and version == 'v1':
        pred = []
        # recursion predict version normal
        for rank in range(1, 13):
            df_copy = _get_lgbm_features_version_1(df)
            df_train = df_copy.loc[df_copy['is_train'] == 1]
            df_test = df_copy.loc[df_copy['is_train'] == 0]

            features = [c for c in df_train.columns if c not in ['pm', 'is_train', 'max',
                                                                 'u_q', 'coolant', 'stator_winding', 'u_d',
                                                                 'stator_tooth',
                                                                 'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient',
                                                                 'torque'
                                                                 ]]
            label = 'pm'

            X_tr_val = df_train.copy()
            X_te = df_test[features].iloc[0]

            y_pred = 0
            cvs = [48, 36, 24]

            for cv in cvs:
                cv -= rank
                print('=' * 10 + str(cv) + '=' * 10)
                train = X_tr_val[X_tr_val['rank'] > cv]
                val = X_tr_val[X_tr_val['rank'] == cv]

                X_train, y_train = train[features], train[label]
                X_valid, y_valid = val[features], val[label]

                lgbm_train = lgbm.Dataset(X_train, y_train)
                lgbm_valid = lgbm.Dataset(X_valid, y_valid)

                model_mae = lgbm.train(params=lgb_params,
                                       train_set=lgbm_train,
                                       valid_sets=[lgbm_train, lgbm_valid],
                                       num_boost_round=3600,
                                       verbose_eval=100, )
                y_pred += model_mae.predict(X_te)
            y_pred = y_pred / len(cvs)

            df['is_train'].iloc[-(13 - rank)] = 1
            df['pm'].iloc[-(13 - rank)] = y_pred
            pred.append(y_pred)

        return pred

    if recursion and version == 'v2':
        pred = []
        # recursion predict version normal
        df_copy = _get_lgbm_features_version_2(df)

        df_train = df_copy.loc[df_copy['is_train'] == 1]
        df_test = df_copy.loc[df_copy['is_train'] == 0]

        features = [c for c in df_train.columns if c not in ['pm', 'is_train', 'rank', 'seq_id']]
        label = 'pm'

        X_tr_val = df_train.copy()
        X_te = df_test[features].iloc[0]

        y_pred = 0
        cvs = [27, 26, 25]
        for cv in cvs:
            print('=' * 10 + str(cv) + '=' * 10)
            for rank in range(1, 13):
                print('=' * 10 + str(rank) + '=' * 10)
                train = X_tr_val[X_tr_val['rank'] > cv]
                val = X_tr_val[X_tr_val['rank'] == cv - rank + 1]

                X_train, y_train = train[features], train[label]
                X_valid, y_valid = val[features], val[label]

                lgbm_train = lgbm.Dataset(X_train, y_train)
                lgbm_valid = lgbm.Dataset(X_valid, y_valid)

                model_mae = lgbm.train(params=lgb_params,
                                       train_set=lgbm_train,
                                       valid_sets=[lgbm_train, lgbm_valid],
                                       num_boost_round=3600,
                                       verbose_eval=100, )
                y_pred += model_mae.predict(X_te)
            y_pred = y_pred / len(cvs)

            df['is_train'].iloc[-(13 - rank)] = 1
            df['pm'].iloc[-(13 - rank)] = y_pred
            pred.append(y_pred)

    df_copy = _get_lgbm_features(df)
    df_copy = df_copy.loc[df_copy['seq_id'] > 31]
    df_train = df_copy.loc[df_copy['is_train'] == 1]
    df_test = df_copy.loc[df_copy['is_train'] == 0]

    features = [c for c in df_train.columns if c not in ['pm', 'is_train', 'max',
                                                         'u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth',
                                                         'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient', 'torque'
                                                         ]]
    label = 'pm'

    X_tr_val = df_train.copy()
    X_te = df_test[features]

    y_pred = 0
    cvs = [48, 36, 24]

    for cv in cvs:
        print('=' * 10 + str(cv) + '=' * 10)
        train = X_tr_val[X_tr_val['rank'] > cv]
        val = X_tr_val[X_tr_val['rank'] <= cv]
        val = val[val['rank'] > (cv - 12)]

        X_train, y_train = train[features], train[label]
        X_valid, y_valid = val[features], val[label]

        lgbm_train = lgbm.Dataset(X_train, y_train)
        lgbm_valid = lgbm.Dataset(X_valid, y_valid)

        model_mae = lgbm.train(params=lgb_params,
                               train_set=lgbm_train,
                               valid_sets=[lgbm_train, lgbm_valid],
                               num_boost_round=3600,
                               verbose_eval=100, )
        y_pred += model_mae.predict(X_te)

    return y_pred / len(cvs)


def main():
    df = get_df()
    train(df)


if __name__ == '__main__':
    main()
'''

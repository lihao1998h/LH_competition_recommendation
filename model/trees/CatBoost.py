from sklearn.model_selection import TimeSeriesSplit
import numpy as np
# from CatBoost import CatBoostRegressor
from sklearn import preprocessing, metrics
# import catboost as cb
features = ['date_block_num',
            'month',
            'shop_id',
            'item_id',
            'city_code',
            'item_category_id',
            'type_code',
            'subtype_code',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
       'price_shift_1', 'price_shift_2', 'price_shift_3', 'price_shift_6',
       'price_shift_12', 'date_cnt_lag_1', 'date_item_lag_1',
       'date_item_lag_2', 'date_item_lag_3', 'date_item_lag_6',
       'date_item_lag_12', 'date_shop_lag_1', 'date_shop_lag_2',
       'date_shop_lag_3', 'date_shop_lag_6', 'date_shop_lag_12',
       'date_cat_lag_1']
cat_features = ['month', 'shop_id','item_id','city_code', 'item_category_id', 'type_code', 'subtype_code']


# test_pred_cat = train_catboost(df)

def train_catboost(df):
    '''train a catboost
    '''
    df.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)
    x_train = df[df['date_block_num'] < 34]
    y_train = x_train['item_cnt_month'].astype(np.float32)
    test = df[df['date_block_num'] == 34]

    folds = TimeSeriesSplit(n_splits=3)  # use TimeSeriesSplit cv
    splits = folds.split(x_train, y_train)
    val_pred = np.zeros(len(x_train))
    test_pred = np.zeros(len(test))
    for fold, (trn_idx, val_idx) in enumerate(splits):
        print(f'Training fold {fold + 1}')

        train_set = x_train.iloc[trn_idx][features]
        y_tra = y_train.iloc[trn_idx]
        val_set = x_train.iloc[val_idx][features]
        y_val = y_train.iloc[val_idx]

        model = CatBoostRegressor(iterations=1500,
                                  learning_rate=0.03,
                                  depth=5,
                                  loss_function='RMSE',
                                  eval_metric='RMSE',
                                  random_seed=42,
                                  bagging_temperature=0.3,
                                  od_type='Iter',
                                  metric_period=50,
                                  od_wait=28)
        model.fit(train_set, y_tra,
                  eval_set=(val_set, y_val),
                  use_best_model=True,
                  cat_features=cat_features,
                  verbose=50)

        val_pred[val_idx] = model.predict(x_train.iloc[val_idx][features])  # prediction
        # test_pred += model.predict(test[features]) / 3 # calculate mean prediction value of 3 models
        print('-' * 50)
        print('\n')
    test_pred = model.predict(test[features])
    val_rmse = np.sqrt(metrics.mean_squared_error(y_train, val_pred))
    print('Our out of folds rmse is {:.4f}'.format(val_rmse))
    return test_pred



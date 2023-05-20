"""
lgb.py
LightGBM
https://www.jianshu.com/p/6b38dc961f9a
"""

# import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
from baselib.utils import mape_loss_func
from setting import args
# from bayes_opt import BayesianOptimization  # 贝叶斯调参
# from hyperopt import hp
import matplotlib.pyplot as plt

from baselib.metrics import xunfei_sales_loss
# import shap


def plot_feature_importances(df):
    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(20, 12))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:50]))),
            df['importance_normalized'].head(50),
            align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:50]))))
    ax.set_yticklabels(df['feature'].head(50))

    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()

    return df


def get_lgb_params():
    lgb_params = {
        'boosting_type': 'gbdt',  # options: gbdt, rf, dart, goss
        'num_leaves': 2 ** 10 - 1,  # max number of leaves in one tree
        'num_boost_round': 5000,
        # 'min_data_in_leaf': 20,  # 叶节点最小值，用于缓解过拟合
        'objective': 'mse',  # todo 试试poisson和tweedie
        # options: regression/mse/rmse, regression_l1/mae, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg
        # 'tweedie_variance_power': 1.1,
        'max_depth': 10,
        'learning_rate': 0.01,
        "feature_fraction": 0.8,  # 默认值为1；指定每次迭代所需要的特征部分
        "bagging_freq": 1,  # k 意味着每 k 次迭代执行bagging,0表示不bagging
        "bagging_fraction": 0.8,  # 默认值为1；指定每次迭代所需要的数据部分，并且它通常是被用来提升训练速度和避免过拟合的。
        "bagging_seed": args.seed,
        "metric": 'mae',  # 常用：mse,mae,rmse,mape  https://zhuanlan.zhihu.com/p/91511706
        "lambda_l1": 0.1,
        "verbosity": -1,  # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug 默认1
        'early_stopping_rounds': 500,
        # 'colsample_bytree': 0.95,
        'seed': args.seed
    }
    return lgb_params


def lgb_train(train_X, train_y, X_val, y_val, test_X, features, cat_features):
    print('lgb start!')

    params = get_lgb_params()

    # outputs
    oof_lgb = np.zeros(len(train_X))
    predictions_val_lgb = np.zeros(len(X_val))
    predictions_test_lgb = np.zeros(len(test_X))

    feature_importance_values = []
    valid_scores = []
    train_scores = []
    best_iterations = []
    kf = False  # 交叉训练
    if kf:
        folds = KFold(5, shuffle=True)
        # folds = TimeSeriesSplit(5)  # 3
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_y)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(train_X.iloc[trn_idx], train_y.iloc[trn_idx], categorical_feature=cat_features)
            val_data = lgb.Dataset(train_X.iloc[val_idx], train_y.iloc[val_idx], categorical_feature=cat_features)

            clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data],
                            verbose_eval=10)  # , feval=eval_func 自定义验证函数

            feature_importance_values.append(clf.feature_importance())
            best_iteration = clf.best_iteration
            oof_lgb[val_idx] = clf.predict(train_X.iloc[val_idx], num_iteration=clf.best_iteration)

            predictions_val_lgb += clf.predict(X_val, num_iteration=clf.best_iteration) / folds.n_splits
            predictions_test_lgb += clf.predict(test_X, num_iteration=clf.best_iteration) / folds.n_splits
            print(clf.best_score)
            valid_score = clf.best_score['valid_1']['l1']
            train_score = clf.best_score['training']['l1']
            valid_scores.append(valid_score)
            train_scores.append(train_score)
            best_iterations.append(best_iteration)

            # explainer = shap.TreeExplainer(clf)
            # shap_df = train_X.sample(100)
            # shap_values = explainer.shap_values(shap_df)
            # shap.summary_plot(shap_values, shap_df, max_display=100)
            # shap.summary_plot(shap_values, shap_df, plot_type="bar", max_display=100)

            gc.enable()
            del clf, trn_data, val_data
            gc.collect()

        fold_names = list(range(folds.get_n_splits()))
        fold_names.append('overall')

        valid_scores.append(np.mean(valid_scores))
        train_scores.append(np.mean(train_scores))

        metrics = pd.DataFrame({'fold': fold_names,
                                'train': train_scores,
                                'valid': valid_scores})

        # 需要其它loss需要自己构造

        feature_importance_values = np.array(feature_importance_values).T

        feature_importances = pd.DataFrame({'feature': features,
                                            'importance': np.mean(feature_importance_values, axis=1)
                                            })
        feature_importances.to_csv('./output/fea_importance.csv', index=False)
        # plot_feature_importances(feature_importances)

    kf_stacking = True
    if kf_stacking:
        print('stacking')
        # step1
        folds = KFold(5, shuffle=True)
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_y)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(train_X.iloc[trn_idx], train_y.iloc[trn_idx], categorical_feature=cat_features)
            val_data = lgb.Dataset(train_X.iloc[val_idx], train_y.iloc[val_idx], categorical_feature=cat_features)

            clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data],
                            verbose_eval=10)  # , feval=eval_func 自定义验证函数

            oof_lgb[val_idx] = clf.predict(train_X.iloc[val_idx], num_iteration=clf.best_iteration)

            predictions_val_lgb += clf.predict(X_val, num_iteration=clf.best_iteration) / folds.n_splits
            predictions_test_lgb += clf.predict(test_X, num_iteration=clf.best_iteration) / folds.n_splits

        # step2
        train_X['stack_lgb'] = oof_lgb
        X_val['stack_lgb'] = predictions_val_lgb
        test_X['stack_lgb'] = predictions_test_lgb

        trn_data = lgb.Dataset(train_X, train_y, categorical_feature=cat_features)

        clf = lgb.train(params, trn_data, valid_sets=[trn_data],
                        verbose_eval=10)

        predictions_val_lgb = clf.predict(X_val, num_iteration=clf.best_iteration)
        predictions_test_lgb = clf.predict(test_X, num_iteration=clf.best_iteration)


    all_to_train = False
    if all_to_train:
        trn_data = lgb.Dataset(train_X, train_y, categorical_feature=cat_features)
        val_data = lgb.Dataset(X_val, y_val, categorical_feature=cat_features)
        clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data], verbose_eval=10)
        predictions_val_lgb = clf.predict(X_val, num_iteration=clf.best_iteration)
        predictions_test_lgb = clf.predict(test_X, num_iteration=clf.best_iteration)

    my_cv = False
    if my_cv:
        cvs = [2018, 2019, 2020]
        for cv in cvs:
            print('=' * 10 + str(cv) + '=' * 10)
            train = train_X.copy()
            train['label'] = train_y
            mask = (train['year'] < cv) | ((train['year'] == cv) & (train['month'] != 12))
            train_df = train[mask]
            val_df = train.loc[train['year'] == cv].loc[train['month'] == 12]

            X_train, y_train = train_df[features], train_df['label']
            X_valid, y_valid = val_df[features], val_df['label']
            trn_data = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
            val_data = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)

            clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data],
                            verbose_eval=10)  # , feval=eval_func 自定义验证函数
            predictions_val_lgb += clf.predict(X_val, num_iteration=clf.best_iteration) / len(cvs)
            predictions_test_lgb += clf.predict(test_X, num_iteration=clf.best_iteration) / len(cvs)
    # oof_lgb_final = np.argmax(oof_lgb, axis=1)
    train_loss = mean_absolute_error(oof_lgb, train_y)
    print("CV train mae loss: {:<8.8f}".format(train_loss))
    val_loss = mean_absolute_error(predictions_val_lgb, y_val)
    print("CV val mae loss: {:<8.8f}".format(val_loss))

    return val_loss, predictions_test_lgb, predictions_val_lgb


def feature_select():
    # https://zhuanlan.zhihu.com/p/32749489
    # 当数据或特征过大也可以使用stacking：https://github.com/liupengsay/2018-Tencent-social-advertising-algorithm-contest/tree/master/%E5%84%BF%E9%A1%BB%E6%88%90%E5%90%8D%E9%85%92%E9%A1%BB%E9%86%89_v2
    def evalsLoss(cols):
        print('Runing...')
        s = time.time()
        clf.fit(train_part_x[:, cols], train_part_y)
        ypre = clf.predict_proba(evals_x[:, cols])[:, 1]
        print(time.time() - s, "s")
        return roc_auc_score(evals_y[0].values, ypre)

    print('开始进行特征选择计算...')
    all_num = int(len(se) / 100) * 100
    print('共有', all_num, '个待计算特征')
    loss = []
    break_num = 0
    for i in range(100, all_num, 100):
        loss.append(evalsLoss(col[:i]))
        if loss[-1] > baseloss:
            best_num = i
            baseloss = loss[-1]
            break_num += 1
        print('前', i, '个特征的得分为', loss[-1], '而全量得分', baseloss)
        print('\n')
        if break_num == 2:
            break
    print('筛选出来最佳特征个数为', best_num, '这下子训练速度终于可以大大提升了')



# 调参lgb，调完再用上面的
def param_tuning(train_X, train_y, folds, metrics, mode='hyper'):
    '''
    mode: bayes / hyper
    '''
    if mode == 'bayes':
        # 给定超参数搜索空间
        params_tuning = {'num_leaves': (10, 100),
                         'num_boost_round': (10, 1000),
                         'learning_rate': (0.005, 0.2),
                         'max_depth': (2, 6),
                         'feature_fraction': (0.5, 1),
                         'bagging_fraction': (0.5, 1),
                         'lambda_l1': (0.01, 0.2),
                         }

        def lgb_param_tuning(num_leaves, num_boost_round, learning_rate, max_depth, feature_fraction, bagging_fraction,
                             lambda_l1):
            # params
            # cat_features = 'auto'
            cat_features = ['product_id', 'type']

            params = {
                'boosting_type': 'gbdt',  # options: gbdt, rf, dart, goss
                'num_leaves': int(num_leaves),  # max number of leaves in one tree
                'num_boost_round': int(num_boost_round),
                'min_data_in_leaf': 20,  # 叶节点最小值，用于缓解过拟合
                'objective': 'rmse',
                # options: regression/mse/rmse, regression_l1/mae, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                "feature_fraction": feature_fraction,  # 默认值为1；指定每次迭代所需要的特征部分
                "bagging_freq": 1,  # k 意味着每 k 次迭代执行bagging,0表示不bagging
                "bagging_fraction": bagging_fraction,  # 默认值为1；指定每次迭代所需要的数据部分，并且它通常是被用来提升训练速度和避免过拟合的。
                "bagging_seed": args.seed,
                "metric": 'rmse',  # 常用：mse,mae,rmse,mape  https://zhuanlan.zhihu.com/p/91511706
                "lambda_l1": lambda_l1,
                "verbosity": -1,  # < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug 默认1
                'early_stopping_rounds': 50,
                'colsample_bytree': 0.95,
                'seed': args.seed
            }

            # outputs
            valid_scores = []
            train_scores = []

            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_y)):
                trn_data = lgb.Dataset(train_X.iloc[trn_idx], train_y.iloc[trn_idx], categorical_feature=cat_features)
                val_data = lgb.Dataset(train_X.iloc[val_idx], train_y.iloc[val_idx], categorical_feature=cat_features)

                clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data],
                                verbose_eval=0)  # , feval=eval_func 自定义验证函数

                valid_score = clf.best_score['valid_1'][metrics]
                train_score = clf.best_score['training'][metrics]
                valid_scores.append(valid_score)
                train_scores.append(train_score)

                gc.enable()
                del clf, trn_data, val_data
                gc.collect()

            val = np.mean(valid_scores)
            return 100 / val

        # 1 贝叶斯调参
        opt = BayesianOptimization(
            lgb_param_tuning,
            params_tuning
        )

        opt.maximize(n_iter=10)  # 最大化黑盒函数
        print(opt.max)  # 返回黑盒函数值最大的超参数
    elif mode == 'grid':
        # 2 随机搜索调参 或 网格搜索调参
        # from sklearn.model_selection import RandomizedSearchCV, or GridSearchCV

        params = {
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [5, 10, 20],
            'min_split_gain': [0, 5, 10],
            'num_leaves': [16, 64, 128],
        }
        model = lgb.LGBMRegressor()
        random_search = RandomizedSearchCV(model, params, cv=5, verbose=2)
        random_search.fit(X_train, y_train)

        print('随机搜索调参得到的最优参数：')
        print(random_search.best_estimator_)
    elif mode == 'hyper':
        def objective(lgb_params):
            '''Objective function for Gradient Boosting Machine Hyperparameter Tuning'''

            # Perform n_fold cross validation with hyperparameters
            # Use early stopping and evalute based on ROC AUC
            cv_results = lgbm.train(lgb_params,
                                    lgbm_train,
                                    num_boost_round=3600,
                                    early_stopping_rounds=400,
                                    valid_sets=[lgbm_train, lgbm_valid],
                                    verbose_eval=100,
                                    feval=xunfei_sales_loss,
                                    feature_name=features
                                    )

            # Extract the best score
            best_score = -cv_results.best_score['valid_1']['x_loss']
            y_pred = cv_results.predict(X_test)

            acc = -x_sales_loss(y_pred, y_test)

            return {'loss': acc, 'params': lgb_params, 'status': STATUS_OK}

        space = {
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'metric': None,
            'n_jobs': -1,
            'seed': 42,
            'bagging_seed': 42,
            'random_state': 42,
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'tweedie_variance_power': hp.uniform('tweedie_variance_power', 1, 2),
            'bagging_freq': hp.randint('bagging_freq', 0, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
            'colsample_bynode': hp.uniform('colsample_bynode', 0, 1),
            'lambda_l1': hp.uniform('lambda_l1', 0, 1),
            'lambda_l2': hp.uniform('lambda_l2', 0, 1),
            'n_estimators': hp.choice('n_estimators', range(50, 300)),
            'bagging_fraction': hp.uniform('bagging_fraction', 0, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0, 1)
        }

        # Algorithm
        tpe_algorithm = tpe.suggest
        # Trials object to track progress
        bayes_trials = Trials()

        MAX_EVALS = 500
        # Optimize
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest,
                           max_evals=MAX_EVALS, trials=bayes_trials)
        print(best_params)

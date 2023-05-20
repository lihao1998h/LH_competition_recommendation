import numpy as np


def xunfei_sales_loss(y_true, y_pred):
    # mape对小值敏感，而官方验证函数对大值敏感
    y_true = y_true.reset_index(drop=True)
    num_product_id = len(y_pred) // 3
    y_true_1, y_pred_1, y_true_2, y_pred_2, y_true_3, y_pred_3 = [], [], [], [], [], []
    for i in range(num_product_id):
        y_true_1.append(y_true[i*3])
        y_pred_1.append(y_pred[i*3])
        y_true_2.append(y_true[i*3+1])
        y_pred_2.append(y_pred[i*3+1])
        y_true_3.append(y_true[i*3+2])
        y_pred_3.append(y_pred[i*3+2])
    acc_1, acc_2, acc_3 = 0.0, 0.0, 0.0

    total_y_true_1 = np.sum(y_true_1)
    total_y_true_2 = np.sum(y_true_2)
    total_y_true_3 = np.sum(y_true_3)

    for i in range(num_product_id):
        ae_1 = np.abs(y_pred_1[i]-y_true_1[i])
        loss_1 = (y_true_1[i] - ae_1) / total_y_true_1
        acc_1 += loss_1

        ae_2 = np.abs(y_pred_2[i]-y_true_2[i])
        loss_2 = (y_true_2[i] - ae_2) / total_y_true_2
        acc_2 += loss_2

        ae_3 = np.abs(y_pred_3[i]-y_true_3[i])
        loss_3 = (y_true_3[i] - ae_3) / total_y_true_3
        acc_3 += loss_3
    print('y_pred_2', y_pred_2)
    print('mean_y_pred_2', np.mean(y_pred_2))
    print('y_true_2', y_true_2)
    print('mean_y_true_2', np.mean(y_true_2))
    acc = (acc_1 + acc_2 + acc_3) / 3
    return acc


def xunfei_sales_loss_feval(y_pred, y_true):
    y_true = y_true.get_label()
    y_pred = y_pred
    num_product_id = len(y_pred) // 3
    y_true_1, y_pred_1, y_true_2, y_pred_2, y_true_3, y_pred_3 = [], [], [], [], [], []
    for i in range(num_product_id):
        y_true_1.append(y_true[i * 3])
        y_pred_1.append(y_pred[i * 3])
        y_true_2.append(y_true[i * 3 + 1])
        y_pred_2.append(y_pred[i * 3 + 1])
        y_true_3.append(y_true[i * 3 + 2])
        y_pred_3.append(y_pred[i * 3 + 2])
    acc_1, acc_2, acc_3 = 0.0, 0.0, 0.0

    total_y_true_1 = np.sum(y_true_1)
    total_y_true_2 = np.sum(y_true_2)
    total_y_true_3 = np.sum(y_true_3)

    for i in range(num_product_id):
        ae_1 = np.abs(y_pred_1[i] - y_true_1[i])
        loss_1 = (y_true_1[i] - ae_1) / total_y_true_1
        acc_1 += loss_1

        ae_2 = np.abs(y_pred_2[i] - y_true_2[i])
        loss_2 = (y_true_2[i] - ae_2) / total_y_true_2
        acc_2 += loss_2

        ae_3 = np.abs(y_pred_3[i] - y_true_3[i])
        loss_3 = (y_true_3[i] - ae_3) / total_y_true_3
        acc_3 += loss_3

    eval_result = (acc_1 + acc_2 + acc_3) / 3
    eval_name = 'x_loss'

    return (eval_name, eval_result, True)

def tweedie_feval(y_pred, y_true, p=1.5):
    y_true = y_true.get_label()
    a = y_true * np.exp(y_pred, (1 - p)) / (1 - p)
    b = np.exp(y_pred, (2 - p)) / (2 - p)
    loss = -a + b
    return loss


def eval_func(preds, dtrain):
    # mape对小值敏感，而官方验证函数对大值敏感
    # not complete
    label = dtrain.get_label()
    preds = preds.reshape(len(label), -1)
    total_preds = np.sum(preds)
    # f1 = f1_score(label, preds, average='weighted')
    mape = mean_absolute_percentage_error(label, preds, multioutput='raw_values')
    acc = np.sum((1 - mape) * (preds / total_preds))
    return 'acc', float(acc), True


# def metric(y_true,y_pred,type=reg):
#     ## type：指标类型：reg回归指标，cls：分类指标，cluster:聚类指标
#     ## y_true：真实值，y_pred：预测值
#
#     from sklearn.metrics import *
#
#     if type==reg:
#         # 回归器评价指标
#
#         mean_squared_error(y_true, y_pred, sample_weight=None, multioutput=’uniform_average’)# 均方误差
#         max_error(y_true, y_pred)# 最大误差
#         mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput=’uniform_average’)# 平均绝对误差
#         #均方根误差
#         # >>explained_variance_score
#         # >mean_squared_log_error
#         # >median_absolute_error
#         # >r2_score
#         # >Mean Absolute Percent Error (MAPE)
#     if type==cls:
#         # 分类器评价指标
#         accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)# 准确率
#         # normalize：默认返回正确率，若为False则返回预测正确的样本数。
#
#         balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)#平衡准确率：每个类上获得的召回率的平均值
#         # adjusted：
#
#         average_precision_score(y_true, y_score, average=’macro’, pos_label=1, sample_weight=None)
#         recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)#（查全率）召回率tp/（tp+fn），直观上是分类器发现所有正样本的能力。
#         precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)#（查准率）精确率tp/（tp+fp），直观地说是分类器不将负样本标记为正样本的能力。
#         f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)#F1实际上是精确率和召回率的调和平均。
#         # average：可指定micro和macro
#
#         #macro-recall/p/f1是先计算再平均，而micro-recall/p/f1是先平均再计算
#
#         log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)#也叫做逻辑损失或交叉熵损失
#         # average_precision_score
#         # brier_score_loss
#         # jaccard_score
#
#         roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None, max_fpr=None)#roc曲线的auc分数
#         classification_report(y_test, credit_pred)#综合报告
#         confusion_matrix(y_test, credit_pred)# 混淆矩阵
#     if type=cluster:
#         # 聚类器评价指标
#         #聚类的好坏不存在绝对标准
#         adjusted_rand_score(labels_true, labels_pred)：ARI指数
#         mutual_info_score(labels_true, labels_pred, contingency=None)：互信息
#         adjusted_mutual_info_score(labels_true, labels_pred, average_method=’warn’)
#         normalized_mutual_info_score(labels_true, labels_pred, average_method=’warn’)
#         completeness_score(labels_true, labels_pred)完备性
#         homogeneity_score(labels_true, labels_pred)：同质性
#         homogeneity_completeness_v_measure(labels_true, labels_pred, beta=1.0)
#         v_measure_score(labels_true, labels_pred, beta=1.0)
#         # 外部指标
#         # Jaccard系数
#         fowlkes_mallows_score(labels_true, labels_pred, sparse=False) # FM指数
#         # Rand指数
#         silhouette_score(X, labels, metric=’euclidean’, sample_size=None, random_state=None, **kwds) 轮廓系数
#         calinski_harabasz_score(X, labels)
#
#
#         # 内部指标
#         davies_bouldin_score(X, labels) # DB指数
#         # Dunn指数，DI
#         contingency_matrix(labels_true, labels_pred, eps=None, sparse=False)





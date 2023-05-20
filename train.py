import os
import time

# from fbprophet import Prophet
from baselib.utils import *
from keras.callbacks import ModelCheckpoint

# 模型
from model.trees.lgb import *
from model.trees.CatBoost import train_catboost

## 模型-时间序列
from model.NN import *
import torch.nn.functional as F


import torch
import torch.nn as nn
from tqdm import tqdm

# 4. model #输入：metrics, data_train, data_test, features 输出：scores,res,模型（feature_importances，best_iterations）
def train_val(model, train_loader, val_loader, args):
    print('-'*20, 'model开始...', '-'*20)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mrr = 0
    for epoch in range(args.num_epoch):
        # train
        print(f'now train epoch: {epoch}')
        print('lr:%.4e' % optimizer.param_groups[0]['lr'])
        model.train()
        for i, (hist_click, target, lens) in enumerate(train_loader):
            hist_click, target = hist_click.to(args.device), target.to(args.device)
            output = model(hist_click, lens)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))


        # val
        model.eval()
        print(f'validation epoch: {epoch}')

        HIT, NDCG, MRR = 0, 0, 0
        length = 0
        for hist_click, target, lens in tqdm(val_loader):
            hist_click, target = hist_click.to(args.device), target.to(args.device)
            output = model(hist_click, lens)
            candidates_score = F.softmax(output[:, 1:], dim=1)
            candidate_argsort = candidates_score.argsort(dim=1, descending=True)
            rec_matrix = candidate_argsort[:, :args.match_num] + 1
            hit, ndcg, mrr = evaluate(rec_matrix, target, args.match_num)
            length += len(rec_matrix)
            HIT += hit
            NDCG += ndcg
            MRR += mrr
        HIT /= length
        NDCG /= length
        MRR /= length
        print('[+] HIT@{} : {}'.format(args.match_num, HIT))
        print('[+] NDCG@{} : {}'.format(args.match_num, NDCG))
        print('[+] MRR@{} : {}'.format(args.match_num, MRR))
        if best_mrr < MRR:
            best_model = model
            best_mrr = MRR
            corr_hit, corr_ndcg = HIT, NDCG
            torch.save(best_model, os.path.join(args.output_root, args.model, f'epoch_{epoch}.pkl'))
            print('Model updated.')

        return best_mrr




    # 时间序列任务的训练
    # 训练集由特征工程构造出cv训练验证集
    X_train, y_train, X_val, y_val = data_train[0], data_train[1], data_train[2], data_train[3]
    X_test = data_test

    model_time_start = time.time()


    features = [c for c in data_test.columns if c not in [args.label]]



    print(f'features count: {len(features)}')
    print(f'features: {features}')


    ## models
    model = 'lgb'
    if model == 'lgb':
        # param_tuning(train_X, train_y, folds, metrics)

        cat_features = ['地区', '站号']

        predictions_test_lgb = np.zeros(X_test.shape[0])
        val_losses = []
        feature_importance_values = []

        print('train_X has {} rows and {} columns'.format(X_train[features].shape[0], X_train[features].shape[1]))
        print('train_y has {} rows'.format(y_train.shape[0]))
        print('X_val has {} rows and {} columns'.format(X_val[features].shape[0], X_val[features].shape[1]))
        print('y_val has {} rows'.format(y_val.shape[0]))
        print('X_test has {} rows and {} columns'.format(X_test.shape[0], X_test.shape[1]))

        val_loss, test_pred_cv, feature_importances = \
            lgb_train(X_train, y_train, X_val, y_val, X_test, features, cat_features)


        # show val scores
        feature_importance_values = np.array(feature_importance_values).T
        feature_importan = pd.DataFrame({'features': features,
                                         'feature_importances': np.mean(feature_importance_values, axis=1)
                                         })

        best_val_scores_df = pd.DataFrame({'cvs': list(range(1, cv_num + 1)),
                                           'val_losses': np.mean(val_losses),
                                           })
        print(feature_importan)
        print(best_val_scores_df)

        test_pred = pd.DataFrame({'商品id': X_test_id,
                               '0': predictions_test_lgb})
    model_time_end = time.time()
    print('='*30, 'model完成，耗时： ', model_time_end - model_time_start, '秒', '='*30)
    return val_loss, test_pred, val_pred



class Loss(nn.Module):
    def __init__(self, reg=0, eps=1e-6):
        super(Loss, self).__init__()
        self.reg = reg
        self.eps = eps

    def forward(self, p, n):
        p = torch.exp(p)
        n = torch.exp(n)
        prob = - torch.log(p / (p + torch.sum(n, dim=1, keepdim=True)) + self.eps)

        return prob.sum() + self.reg


def evaluate(rec_matrix, targets, match_num):
    target_repeats = torch.repeat_interleave(targets.view(-1, 1), dim=1, repeats=match_num)
    judge = torch.where(rec_matrix - target_repeats == 0)  # 分别返回行列True索引
    hit = len(judge[0])
    mrr = 0
    ndcg = 0
    for pos in judge[1]:
        mrr += 1 / (pos.float() + 1)
        ndcg += 1 / torch.log2(pos.float() + 2)

    return hit, ndcg, mrr



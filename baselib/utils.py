import numpy as np
import pandas as pd
import os
# from chinese_calendar import is_workday
import datetime

def check_path(path):
    # 最好保证你的path不是以\\或者/结尾，否则可能产生副作用

    _path = os.path.dirname(path)
    # _path = path
    # 该方法可以剥离路径中的最后一项(若路径以\\或者/结尾则不会剥离)

    if not os.path.exists(_path):
        # 检查该路径是否存在

        os.makedirs(_path, exist_ok=True)
        # 不存在则递归创建该路径，注意exist_ok需要为True

        print(_path, 'created')
        # 提示


# losses
def mape_loss_func(preds, labels):
    mask = labels!=0
    return np.fabs((labels[mask]-preds[mask])/labels[mask]).mean()

# 时间序列
def groupby_shift(df, col, groupcol, shift_n, fill_na = np.nan):
    '''
    apply fast groupby shift
    df: data
    col: column need to be shift
    shift: n
    fill_na: na filled value
    ???
    '''
    tp = df.groupby(groupcol)
    rown = tp.size().cumsum()
    rowno = list(df.groupby(groupcol).size().cumsum()) # 获取每分组第一个元素的index
    lagged_col = df[col].shift(shift_n) # 不分组滚动
    na_rows = [i for i in range(shift_n)] # 初始化为缺失值的index
    #print(na_rows)
    for i in rowno:
        if i == rowno[len(rowno)-1]: # 最后一个index直接跳过不然会超出最大index
            continue
        else:
            new = [i + j for j in range(shift_n)] # 将每组最开始的shift_n个值变成nan
            na_rows.extend(new) # 加入列表
    na_rows = list(set(na_rows)) # 去除重复值
    na_rows = [i for i in na_rows if i <= len(lagged_col) - 1] # 防止超出最大index
    #print(na_rows)
    lagged_col.iloc[na_rows] = fill_na # 变成nan
    return lagged_col




class BayesianSmoothing(object):
    # 这个慢一点
    '''sample
    bs = BayesianSmoothing(1, 1)
    bs.update(temp[feat_1 + '_all'].values, temp[feat_1 + '_1'].values, 1000, 0.001)  # 分别为特征的count和特征的sum
    temp[feat_1 + '_smooth'] = (temp[feat_1 + '_1'] + bs.alpha) / (temp[feat_1 + '_all'] + bs.alpha + bs.beta)

    X_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
    X_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
    #类别少，不用平滑
    '''
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


class HyperParam(object):#平滑，这个快一点；hyper=HyperParam(1, 1); hyper.update_from_data_by_moment(show, click)
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)

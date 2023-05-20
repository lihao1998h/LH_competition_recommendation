import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

# 时序数据
from statsmodels.graphics.tsaplots import plot_pacf


def plot_TS(df, date_col, show_cols, hue_col):
    # df = df.set_index(date_col)
    df_ = df.copy()

    filter = True
    if filter:
        only_month_12 = False
        if only_month_12:
            df_['month'] = df_['date'].dt.month
            df_ = df_[df_['month'] == 12].reset_index(drop=True)
            df_ = df_.drop('month', axis=1)

        only_year_2021 = False
        if only_year_2021:
            df_['year'] = df_['date'].dt.year
            df_ = df_[df_['year'] == 2021].reset_index(drop=True)
            df_ = df_.drop('year', axis=1)

        only_2021_11 = False
        if only_2021_11:
            df_['year'] = df_['date'].dt.year
            df_['month'] = df_['date'].dt.month
            df_ = df_[df_['year'] == 2021].reset_index(drop=True)
            df_ = df_[df_['month'] == 11].reset_index(drop=True)
            df_ = df_.drop(['month', 'year'], axis=1)

        only_2020_11 = True
        if only_2020_11:
            df_['year'] = df_['date'].dt.year
            df_['month'] = df_['date'].dt.month
            df_ = df_[df_['year'] == 2020].reset_index(drop=True)
            df_ = df_[df_['month'] == 11].reset_index(drop=True)
            df_ = df_.drop(['month', 'year'], axis=1)



    unique_hue = df_[hue_col].unique().tolist()
    show_num = 1
    for i in range(len(unique_hue) // show_num):
        # 每次显示show_num条
        df_plot = df_[df_[hue_col].isin([unique_hue[i*show_num+j] for j in range(show_num)])]
        df_plot = df_plot.drop_duplicates()
        # 折线图
        # fig = plt.figure()
        plt.figure(dpi=300, figsize=(24, 8))
        plt.title('Interesting Graph')
        plt.xticks(fontsize=10)
        for show_col in show_cols:
            plt.plot(df_plot[date_col], df_plot[show_col])
        plt.legend(show_cols)
        plt.show()

    # 偏自回归系数 for lagging
    # df = df[df[hue_col].isin([1001])]
    # df_idx = df.set_index(date_col)
    # _ = plot_pacf(df_idx[show_col], lags=12)
    # plt.show()

def plot(X,y,X_cols,y_col,plot_type='scatter'):
    #scatter:散点图，pairplot:sns.pairplot，box：箱图，hist：直方图，heatmap：相关分析，热度图，list:列表汇总
    if plot_type=='scatter':
        # 散点图（num+num）
        data = pd.concat([X, y], axis=1)
        data.plot.scatter(x=X_cols, y=y_col);
    if plot_type=='pairplot':
        #sns.pairplot
        sns.set()
        columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
        sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
        """
        参数：
        size=2.5表示大小，
        aspect=0.8表示，
        kind='reg'添加拟合直线和95%置信区间 'scatter'表示散点图
        """
        plt.show();
    if plot_type=='box':
        # 箱图（num+clas）
        var = 'region'
        data = pd.concat([df[y_col], df[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y=y_col, data=data)
        # fig.axis(ymin=0, ymax=800000);
    if plot_type=='hist':
        # 对比，直方图
        g = sns.FacetGrid(train_df, col='Survived') # row='Pclass', size=2.2, aspect=1.6
        g.map(plt.hist, 'Age', bins=20)
    if plot_type=='heatmap':
        # 相关分析，热度图heatmaps1
        corrmat = df_train.corr()
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);
        # 选出和目标变量最相关的k个变量
        k = 10 #number of variables for heatmap
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(df_train[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
    if plot_type=='list':
        # 列表汇总（分类变量+数字变量）
        train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


## 5.3 动图制作
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# import matplotlib.pyplot as plt
# from matplotlib import animation


def barlist(n):
    taxiorder2019 = pd.read_csv(paths[n], nrows=None,
                                dtype={
                                    'GETON_LONGITUDE': np.float32,
                                    'GETON_LATITUDE': np.float32,
                                    'GETOFF_LONGITUDE': np.float32,
                                    'GETOFF_LATITUDE': np.float32,
                                    'PASS_MILE': np.float16,
                                    'NOPASS_MILE': np.float16,
                                    'WAITING_TIME': np.float16
                                })

    taxiorder2019['GETON_DATE'] = pd.to_datetime(taxiorder2019['GETON_DATE'])
    taxiorder2019['GETON_Hour'] = taxiorder2019['GETON_DATE'].dt.hour

    return taxiorder2019.groupby(['GETON_Hour'])['PASS_MILE'].mean().values


def animate(i):
    print(i)
    y = barlist(i + 1)
    for idx, b in enumerate(barcollection):
        b.set_height(y[idx])
    plt.ylim(0, 8)

    print(i + 1)
    plt.title(paths[i + 1].split('/')[-1])
    plt.ylabel('PASS_MILE / KM')
    plt.xlabel('Hour')

# anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n-1,
#                              interval=500)
#
# anim.save('order.gif', dpi=150)

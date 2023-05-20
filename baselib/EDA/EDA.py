import time
from baselib.EDA.EDA_Base import easy_look, single_variable_EDA, one_key_EDA, reduce_mem_usage
# 自定义的预处理
from baselib.Data_Processing.data_preprocess import data_preprocess
from baselib.EDA.EDA_plot import plot_TS


# 1 EDA1 输入：数据文件 输出：df_train,df_test
# 探索性数据分析(Exploratory Data Analysis,EDA1)
def EDA(args):
    print('='*30, 'EDA开始', '='*30)




    if args.is_reduce_mem_usage:
        print('-'*20, '减少数据大小开始...', '-'*20)
        df_train = reduce_mem_usage(df_train)
        df_test = reduce_mem_usage(df_test)
        print('-'*20, '减少数据大小完成', '-'*20)

    args.train_test_same = True

    if args.EDA:
        print('先看一下训练集和测试集分布是否类似（todo)')
        train_test_same = True
        if args.task_type == 'category' or args.task_type == 'regression':
            pass
        if args.task_type == 'time_series_pred':
            show_cols = [args.label, '湿球空气温度', '露点空气温度', '蒸气压']
            plot_TS(df_train, date_col=args.time_series, show_cols=show_cols, hue_col=args.class_name)

        if args.Auto_EDA:
            print('一键生成EDA中...')
            one_key_EDA(df_train, output_name="./output/train_EDA.html")
            one_key_EDA(df_test, output_name="./output/test_EDA.html")
            print('一键EDA生成完毕...')

        print('------------------整体看一下训练集-----------------')
        args.is_dup, args.is_missing = easy_look(df_train)

        print('-----------------开始单变量分析----------------------')

        single_variable_EDA(df_train, args.label, args.task_type)
    else:
        args.is_dup = False
        args.is_missing = True
        args.is_same_distribution = True

    print('='*30, 'EDA结束', '='*30)
    return df_train, df_test

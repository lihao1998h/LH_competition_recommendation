from matplotlib import pyplot as plt
import os
import pandas as pd


def plot_fig(df, product_id, loss, args):
    # df_train = df.loc[df['model'] == 'train'].loc[df['session_id'] == product_id]
    # df_lgb = df.loc[df['model'] == 'lgb'].loc[df['session_id'] == product_id]
    # df_arima = df.loc[df['model'] == 'arima'].loc[df['session_id'] == product_id]
    # df_arima_13_24 = df.loc[df['model'] == 'arima_13_24'].loc[df['session_id'] == product_id]
    # df_submit = df.loc[df['model'] == 'submit'].loc[df['session_id'] == product_id]

    fig = plt.figure(figsize=(25, 10))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)

    name_list = ['train_2020_12', 'val_2020_12', 'lgb_2021_12']
    for name in name_list:
        df_copy = df.loc[df['model'] == name].copy()
        df_copy = df_copy.loc[df_copy[args.class_name] == product_id, ['日', args.label]]
        plt.plot(df_copy['日'], df_copy[args.label])

        # df_copy2 = df.loc[df['model'] == name].copy()
        # df_copy2 = df_copy2.loc[df_copy2['日'] < 50]
        # df_copy2 = df_copy2.loc[df_copy2[args.class_name] == product_id, ['日', args.label]]
        # ax2.plot(df_copy2['日'], df_copy2[args.label])

    plt.legend(name_list, title=str(loss))

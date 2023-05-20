import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

def test_and_submit(model, test_loader, args):
    model.eval()
    print('testing')
    productfile = pd.read_csv(os.path.join(args.data_root, 'products_train.csv'))  # 商品信息
    task = 'task1'
    test_data = pd.read_csv(os.path.join(args.data_root, f'sessions_test_{task}.csv'))
    predictions = test_data.copy()
    predictions.loc[:, 'next_item_prediction'] = 0
    idx2product = dict(zip(range(1, productfile['id'].nunique() + 1), productfile['id'].unique()))

    with torch.no_grad():
        for i, (hist_click, lens) in enumerate(tqdm(test_loader)):
            hist_click = hist_click.to(args.device)
            output = model(hist_click, lens)
            candidates_score = F.softmax(output[:, 1:], dim=1)
            candidate_argsort = candidates_score.argsort(dim=1, descending=True)
            rec_matrix = candidate_argsort[:, :args.match_num] + 1
            for j in range(len(rec_matrix)):
                next_pred = [idx2product[int(element)] for element in rec_matrix[j]]
                predictions['next_item_prediction'].iloc[i*args.batch_size+j] = next_pred


    predictions.drop('prev_items', inplace=True, axis=1)
    predictions.reset_index(drop=True)

    check_predictions(test_data, productfile, predictions)
    # Its important that the parquet file you submit is saved with pyarrow backend
    predictions.to_parquet(os.path.join(args.output_root, args.model, f'submission_{task}.parquet'), engine='pyarrow')



def check_predictions(test_data, productfile, predictions, check_products=False):
    """
    These tests need to pass as they will also be applied on the evaluator
    """
    test_locale_names = test_data['locale'].unique()
    for locale in test_locale_names:
        sess_test = test_data.query(f'locale == "{locale}"')
        preds_locale = predictions[predictions['locale'] == sess_test['locale'].iloc[0]]
        assert sorted(preds_locale.index.values) == sorted(
            sess_test.index.values), f"Session ids of {locale} doesn't match"

        if check_products:
            # This check is not done on the evaluator
            # but you can run it to verify there is no mixing of products between locales
            # Since the ground truth next item will always belong to the same locale
            # Warning - This can be slow to run
            products = productfile.query(f'locale == "{locale}"')
            predicted_products = np.unique(np.array(list(preds_locale["next_item_prediction"].values)))
            assert np.all(np.isin(predicted_products, products['id'])), f"Invalid products in {locale} predictions"
    print('checked fine!')


def test():





    # 融合 + 提交
    df_train = pd.read_csv('./data/地区温度预测挑战赛公开数据/train.csv', index_col=None)
    # df_train_2021_11 = df_train.loc[df_train['年'] == 2021].loc[df_train['月'] == 11]
    # df_train_2021_10 = df_train.loc[df_train['年'] == 2021].loc[df_train['月'] == 10]
    df_train_2020_12 = df_train.loc[df_train['年'] == 2020].loc[df_train['月'] == 12]
    # df_train_2019_12 = df_train.loc[df_train['年'] == 2019].loc[df_train['月'] == 12]
    pred_val = pd.read_csv('output/pred_val.csv', header=0)

    # pred_arima = pd.read_csv('output/arima-all-base.csv', header=0)
    # pred_arima_13_24 = pd.read_csv('output/arima-13-24.csv', header=0)
    # pred_arima_13_24['rank'] = pred_arima_13_24['rank'] + 12
    pred_lgb = pd.read_csv('output/pred_lgb.csv', header=0)  # 0.08819
    # pred_rule1 = pd.read_csv('output/pred_rule1.csv', header=0)  # 3.53
    # pred_prophet = pd.read_csv('output/pred_prophet.csv', header=0)  # 2.21531

    # arima 和 lgb的融合 ： 分id各取预测
    # df_submit = pred_lgb.copy()
    # for id in pred_lgb['session_id'].unique():
    # if dict[id][3] == 1:
    #     df_submit.loc[df_submit['session_id'] == id, 'pm'] = pred_arima.loc[df_submit['session_id'] == id, 'pm'].values
    # elif dict[id][3] == 2:
    #     df_submit.loc[df_submit['session_id'] == id, 'pm'] = pred_lgb.loc[df_submit['session_id'] == id, 'pm'].values
    # elif dict[id][3] == 3:
    #     df_submit.loc[df_submit['session_id'] == id, 'pm'] = 0.5 * pred_lgb.loc[df_submit['session_id'] == id, 'pm'].values\
    #                                                          + 0.5 * pred_arima.loc[df_submit['session_id'] == id, 'pm'].values
    # elif dict[id][3] == 4:
    # df_submit.loc[df_submit['session_id'] == id, 'pm'] = 0.3 * pred_lgb.loc[df_submit['session_id'] == id, 'pm'].values\
    #                                                          + 0.7 * pred_arima.loc[df_submit['session_id'] == id, 'pm'].values

    # df_submit.to_csv(f'output/submit_lgb.csv', index=None)

    # val arima loss
    # df_true = df_train.loc[df_train['rank'] < 25]
    # val_mse = []
    # for i in range(295):
    #     true = df_true.loc[df_true['session_id'] == i, 'pm'].values
    #     pred = pred_arima_13_24.loc[pred_arima_13_24['session_id'] == i, 'pm'].values
    #     val_mse.append(mean_absolute_error(true, pred))
    # print(val_mse)
    # ill_loss = [x for x in val_mse if x > 0.3]
    # idx = [val_mse.index(x) for x in ill_loss]
    #
    # ill_id = pd.DataFrame({'session_id': idx,
    #                        'val_loss': ill_loss
    #                        })
    # print("Validation mae:", np.mean(val_mse))
    # print(ill_id)

    # plot
    # df_train_2021_11['model'] = 'train_2021_11'
    # df_train_2021_10['model'] = 'train_2021_10'
    df_train_2020_12['model'] = 'train_2020_12'
    # df_train_2019_12['model'] = 'train_2019_12'
    pred_val['model'] = 'val_2020_12'
    pred_lgb['model'] = 'lgb_2021_12'
    # pred_prophet['model'] = 'prophet_2021_12'
    # pred_arima['model'] = 'arima'
    # pred_arima_13_24['model'] = 'arima_13_24'
    # df_submit['model'] = 'submit_2021_12'
    # df_submit_8_2['model'] = '0.8airma0.2lgb'
    df_plot = pd.concat([df_train_2020_12, pred_val, pred_lgb])
    df_plot = df_plot.fillna(0)

    save_path = './output/final' + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for id in tqdm(df_plot[args.class_name].unique()):
        y_true = df_train_2020_12.loc[df_train_2020_12['站号'] == id]['气温']
        y_pred = pred_val.loc[pred_val['站号'] == id]['气温']

        loss = mean_absolute_error(y_true, y_pred)
        plot_fig(df_plot, id, loss, args)

        plt.savefig(
            save_path + str(id) + '.png')
        plt.close()

        print(f'save image id {id} finished')
    print('save image finished')
    # 5. submission #输入：prediction_df 输出: submission.csv

    # # prediction_df = pd.read_csv('./special/predictions.csv')
    # prediction_df = pd.read_csv('special/pre.csv')
    # meter_with_logic_df = pd.read_csv('special/meter_with_logic.csv')
    # output_df = pd.read_csv('special/output_data.csv')
    # submission_df = pd.DataFrame()
    # no_area_meters = [11530, 11532, 11535, 11536, 11410, 11411, 11296, 11297, 10918, 11443, 11444, 11327, 11201,
    #                     11202, 11329, 10831, 10832, 10961,
    #                     11087, 11088, 11220, 11221, 60001, 60002, 60003, 60004, 60005, 60006, 60007, 11245, 11246,
    #                     11001, 11003, 11004, 11006]
    #
    # pred_date = dateRange("2022/04/18", "2022/04/25")
    # # meter to logic
    # with tqdm(total=len(all_data[args.ID].unique())) as pbar:
    #     for ID in all_data[args.ID].unique():
    #         pbar.update(1)
    #         if ID not in no_area_meters:
    #             IDs_logic = meter_with_logic_df[meter_with_logic_df['meter'] == ID]['c_logic_id'].values
    #             for i in range(args.n_output_step):
    #                 value = prediction_df[str(ID) + '.0'].values.tolist()[17+i]
    #                 output_df.loc[output_df['c_logic_id'] == IDs_logic[0], pred_date[i]] += value
    #
    # # logic_sum_up
    # # 当前策略：如果father已经有数据了就不加，否则向上加
    # submission_df = logic_sum_up(output_df, pred_date)
    # submission_df.to_csv('./special/hhu-1fea_4bz_100epoch-0508.csv', index=False, encoding='utf-8')

from special.layout_display import logic_sum_up
import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm

data_to_baseline = pd.read_csv('special/data_to_baseline.csv', header=0)

unique_meter_list = data_to_baseline['meter'].unique()
def dateRange(start, end, step=1, format="%Y/%m/%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]

prediction_df = pd.read_csv('special/predictions_prophet.csv')
for col in prediction_df.columns:
    prediction_df[col] = np.where(prediction_df[col] < 0, 0, prediction_df[col])  # 小的数改为0
meter_with_logic_df = pd.read_csv('special/meter_with_logic.csv')

output_df = pd.read_csv('special/output_data.csv')
submission_df = pd.DataFrame()
no_area_meters = [11530, 11532, 11535, 11536, 11410, 11411, 11296, 11297, 10918, 11443, 11444, 11327, 11201,
                  11202, 11329, 10831, 10832, 10961,
                  11087, 11088, 11220, 11221, 60001, 60002, 60003, 60004, 60005, 60006, 60007, 11245, 11246,
                  11001, 11003, 11004, 11006]

pred_date = dateRange("2022/04/18", "2022/04/25")
# meter to logic
with tqdm(total=len(unique_meter_list)) as pbar:
    for ID in unique_meter_list:
        pbar.update(1)
        if ID not in no_area_meters:
            IDs_logic = meter_with_logic_df[meter_with_logic_df['meter'] == ID]['c_logic_id'].values
            for i in range(7):
                value = prediction_df[str(ID)].values.tolist()[i]
                output_df.loc[output_df['c_logic_id'] == IDs_logic[0], pred_date[i]] += value

# logic_sum_up
# 当前策略：如果father已经有数据了就不加，否则向上加
submission_df = logic_sum_up(output_df, pred_date)
submission_df.to_csv('./special/hhu-5-0511.csv', index=False, encoding='utf-8')
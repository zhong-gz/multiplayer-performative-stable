import pandas as pd
import numpy as np

def read_data(path = 'CournotCompetition/Global Crude Petroleum Trade 1995-2021.csv'):
    df = pd.read_csv(path)
    df_2021 = df[df['Year'] == 2021]
    trade_pivot = df_2021.pivot_table(
        index='Country', 
        columns='Action', 
        values='Trade Value', 
        aggfunc='sum', 
        fill_value=0
    )
    export_gt_import = trade_pivot[trade_pivot['Export'] > (trade_pivot['Import']+68.22*10000)].copy()
    result = pd.merge(
        export_gt_import.reset_index(),
        df_2021[['Country', 'Continent']].drop_duplicates(),
        on='Country',
        how='left'
    )
    export_gt_import['Net Export'] = export_gt_import['Export'] - export_gt_import['Import']
    export_gt_import['Crude Oil Volume (Barrels)'] = export_gt_import['Net Export'] / 68.22

    # 创建只包含国家和原油数量的新数据框
    global_oil_volume = pd.DataFrame({
        'Country': export_gt_import.index,
        'Crude Oil Volume (Barrels)': export_gt_import['Crude Oil Volume (Barrels)']
    })
    global_oil_volume['Crude Oil Volume (Barrels)'] = global_oil_volume['Crude Oil Volume (Barrels)'].astype(np.int64)
    # print("2021年出口额大于进口额的全球国家原油净出口量（桶）:")
    # print(global_oil_volume)
    return global_oil_volume
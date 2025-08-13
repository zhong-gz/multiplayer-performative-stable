import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.ticker import EngFormatter
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1,'./utils/' )
# from utils.utilsrm import *
import time

df = pd.read_csv('CournotCompetition/Global Crude Petroleum Trade 1995-2021.csv')
df_2021 = df[df['Year'] == 2021]
trade_pivot = df_2021.pivot_table(
    index='Country', 
    columns='Action', 
    values='Trade Value', 
    aggfunc='sum', 
    fill_value=0
)
export_gt_import = trade_pivot[trade_pivot['Export'] > trade_pivot['Import']]
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
print("2021年出口额大于进口额的全球国家原油净出口量（桶）:")
print(global_oil_volume)



# avarage of crude oil price is 68.22 per barel in 2021 according to WTI data

# ## Initialize the game class and set the random seed and initial point
# # seed 
# seed = 42
# np.random.seed(seed)
# num_experiments = 10
# figuresize=(25, 6)
# loc_cap=11
# eta=0.001
# run_experiment = 1 # 1: run the experiment, 0: load the data
# mu_A_list = [0.25,0.50,0.75,1.0] #25,0.50,0.75,1
# subrange = 26
# gamma = 2.1
# BATCH=10
# MAXITER=1000
# tot_rev=1

# # 不同的模型
# fs=40
# lw=4
# lw2 = lw/2
# models = ['SIR$^2$', 'RGD','SFB','AGM','OPGD']
# companies = ['Lyft', 'Uber']
# info_types = ['price','rev', 'demand']
# # 定义不同模型对应的颜色
# style_dict = {
#     'SIR$^2$': {'color': '#FF7F50', 'linestyle': '-', 'linewidth': lw+1},
#     'AGM': {'color': '#9467bd', 'linestyle': '--', 'linewidth': lw},
#     'RGD': {'color': '#444444', 'linestyle': ':', 'linewidth': lw},
#     'SFB': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': lw},
#     'OPGD': {'color': '#1f77b4', 'linestyle': (0, (5, 5)), 'linewidth': lw}
# }
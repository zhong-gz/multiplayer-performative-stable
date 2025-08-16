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
from sklearn.preprocessing import MinMaxScaler
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1,'./utils/' )
from CournotFunction import *
import time

global_oil_volume = read_data('CournotCompetition/Global Crude Petroleum Trade 1995-2021.csv')
data = global_oil_volume.to_numpy()
data=data[:, 1]
data = np.partition(data, -28)[-28:]
n = np.size(data)
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_2d = data.reshape(-1, 1)
# scaled = scaler.fit_transform(data_2d)
# data = scaled.flatten()
# print(max(data))
# print(min(data))
# print(np.mean(data))
# print(np.median(data))
# print(np.sum(data))
# print(79.05/np.sum(data))
# print(n)

# 68.22 is the average price of crude oil per barrel in 2021 according to WTI data

# Initialize the game class and set the random seed and initial point
seed = 42
np.random.seed(seed)
num_experiments = 10 #10
figuresize=(25, 6)
loc_cap=11
run_experiment = 1 # 1: run the experiment, 0: load the data
subrange = 26
c_alg = 2.1
tot_rev=1

# 不同的模型
fs=40
lw=4
lw2 = lw/2
models = ['SIR$^2$','RR', 'RGD']#,'SFB','AGM','OPGD']
info_types = ['quantity','revenue','price']
# 定义不同模型对应的颜色
style_dict = {
    'SIR$^2$': {'color': '#FF7F50', 'linestyle': '-', 'linewidth': lw+1},
    'RR': {'color': "#9b0000", 'linestyle': (0, (5, 5)), 'linewidth': lw},
    'AGM': {'color': '#9467bd', 'linestyle': '--', 'linewidth': lw},
    'RGD': {'color': '#444444', 'linestyle': ':', 'linewidth': lw},
    'SFB': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': lw},
    'OPGD': {'color': '#1f77b4', 'linestyle': (0, (5, 5)), 'linewidth': lw}
}

all_data={}
# gamma=[1.0,1.0]
gamma=[0.1,0.1]
c = 10 # cost of oil
p = 68.22 # average price of oil in 2021
z0 = 147.27 # highest oil price in 2008.07
b = (z0 - p)/np.sum(data) # linear demand coefficient
MAXITER=100
eta=0.001
mu_A_list = [0.25,0.50,0.75,1.0] #25,0.50,0.75,1

all_mu_A_data = {}
total_revenue_stats = []
all_mu_A_data_path = 'CournotCompetition/data/all_mu_data.npy'
if run_experiment:
    for mu_A in mu_A_list:
        print('Runing at mu_A',mu_A)
        total_avg_data = {}
        total_var_data = {}
        all_p_data = {}
        data_file_path = f'CournotCompetition/data/mu_A_{mu_A}_data.npy'        
        all_data={}
        for num_exper in range(num_experiments):
            seed = seed+1
            np.random.seed(seed)
            params = {}
            print('    Runing at number',num_exper+1,'trail')
            x0=(np.random.rand(n)*0.2-0.1)*data # initial point in the range [0,5]
            all_data[num_exper]={}
            all_data[num_exper]['x0']=x0

            dic_data = []
            dic_data.append(runSIRR(z0,data,c,b,c_alg, MAXITER,mu_A))
            dic_data.append(runRR(z0,data,c,b, MAXITER,mu_A))
            dic_data.append(runRGD(z0,data,c,b, MAXITER,mu_A,eta,x0))

            for model, dic in zip(models, dic_data):
                # 从字典中获取 x 数据
                all_data[num_exper][f'x_{model}'] = dic['x']

                all_data[num_exper][f'{info_types[0]}_{model}'] = dic['quantity_total']
                all_data[num_exper][f'{info_types[1]}_{model}'] = dic['revenue_total']
                all_data[num_exper][f'{info_types[2]}_{model}'] = dic['price']

        avg_data = {}
        var_data = {}
        for model in models:
            for info_type in info_types:
                key = f'{info_type}_{model}'
                all_values = [all_data[num_exper][key] for num_exper in range(num_experiments)]
                all_values_arr = np.asarray(all_values)
                avg_data[key] = np.mean(all_values_arr, axis=0)
                var_data[key] = np.var(all_values_arr, axis=0)  # 计算方差

        # 为当前p值创建一个结果字典，包含avg_data和var_data
        p_result = {
            'avg': avg_data,
            'var': var_data
        }
        
        # 将当前p值的结果存入主字典
        all_p_data[p] = p_result

        for key, value in avg_data.items():
            if key in total_avg_data:
                total_avg_data[key] = total_avg_data[key] + value.copy()
            else:
                total_avg_data[key] = value.copy()

        for key, value in var_data.items():
            if key in total_var_data:
                total_var_data[key] = total_var_data[key] + value.copy()
            else:
                total_var_data[key] = value.copy()

        all_mu_A_data[mu_A] = {'avg': total_avg_data, 'var': total_var_data,'p_data': all_p_data}  
    np.save(all_mu_A_data_path, all_mu_A_data)
else:
    all_mu_A_data = np.load(all_mu_A_data_path, allow_pickle=True).item()
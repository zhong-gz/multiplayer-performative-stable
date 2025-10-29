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
import matplotlib.ticker as mticker

global_oil_volume = read_data('CournotCompetition/Global Crude Petroleum Trade 1995-2021.csv')
data = global_oil_volume.to_numpy()
data=data[:, 1]
data = np.partition(data, -28)[-28:]
n = np.size(data)

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
models = ['SIR$^2$','RR','RGD','SFB','AGM','OPGD']
info_types = ['quantity','revenue','price']
# 定义不同模型对应的颜色
style_dict = {
    'SIR$^2$': {'color': '#FF7F50', 'linestyle': '-', 'linewidth': lw+1},
    'RR': {'color': "#9b0000", 'linestyle': (0, (2, 2)), 'linewidth': lw},
    'AGM': {'color': '#9467bd', 'linestyle': '--', 'linewidth': lw},
    'RGD': {'color': '#444444', 'linestyle': ':', 'linewidth': lw},
    'SFB': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': lw},
    'OPGD': {'color': '#1f77b4', 'linestyle': (0, (5, 5)), 'linewidth': lw}
}

all_data={}
c = 10 # cost of oil
p = 68.22 # average price of oil in 2021
z0 = 147.27 # highest oil price in 2008.07
b =(z0 - p)/np.sum(data) # linear demand coefficient
MAXITER=100 #1000 for time comparison, 100 for revenue comparison
eta=0.001
k = 1
mu_A_list = [0.25,0.50,0.75,1.0] #25,0.50,0.75,1
mu_A_list = [x * k for x in mu_A_list]
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
            x0=(np.random.rand(n)*0.3)*np.mean(data)  # initial point in the range [-0.5,0.5] -0.5
            all_data[num_exper]={}
            all_data[num_exper]['x0']=x0

            dic_data = []
            dic_data.append(runSIRR(z0,data,c,b,c_alg, MAXITER,mu_A))
            dic_data.append(runRR(z0,data,c,b, MAXITER,mu_A))
            dic_data.append(runRGD(z0,data,c,b, MAXITER,mu_A,eta,x0))
            dic_data.append(runSFB(z0,data,c,b, MAXITER,mu_A,eta,x0))
            dic_data.append(runAGM(z0,data,c,b, MAXITER,mu_A,eta,x0))
            dic_data.append(runOPGD(z0,data,c,b, MAXITER,mu_A,eta,x0))

            for model, dic in zip(models, dic_data):
                # 从字典中获取 x 数据
                all_data[num_exper][f'x_{model}'] = dic['x'] #adjustment
                all_data[num_exper][f'{info_types[0]}_{model}'] = dic['quantity_total']
                all_data[num_exper][f'{info_types[1]}_{model}'] = dic['revenue_total']
                all_data[num_exper][f'{info_types[2]}_{model}'] = dic['price']
                all_data[num_exper][f'iteration_times_{model}'] = dic['iteration_times']

        avg_data = {}
        var_data = {}
        for model in models:
            for info_type in info_types:
                key = f'{info_type}_{model}'
                all_values = [all_data[num_exper][key] for num_exper in range(num_experiments)]
                all_values_arr = np.asarray(all_values)
                avg_data[key] = np.mean(all_values_arr, axis=0)
                var_data[key] = np.var(all_values_arr, axis=0)  # 计算方差
            # 添加时间数据的平均
            time_key = f'iteration_times_{model}'
            all_time_values = [all_data[num_exper][time_key] for num_exper in range(num_experiments)]
            all_time_values_arr = np.asarray(all_time_values)
            avg_data[time_key] = np.mean(all_time_values_arr, axis=0)

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

## Plotting revenue
fig, axes = plt.subplots(1, 4, figsize=figuresize)
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    for model in models:    
        key1 = f'{info_types[1]}_{model}'
        all_y_data.extend(total_avg_data[key1])
all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(all_y_data))))
scale_factor = 10 ** power
y_min = 0 #min(all_y_data)/ scale_factor -0.2
y_max = max(all_y_data)/ scale_factor +0.2
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    total_var_data = all_mu_A_data[mu_A]['var']
    # ax = fig.add_subplot(gs[0, i])
    ax = axes[i]
    for model in models:
        key1 = f'{info_types[1]}_{model}'
        style = style_dict[model]
        # color = model_colors[model]
        # alpha = company_styles[company]['alpha']
        total_rev = (total_avg_data[key1])/ scale_factor
        total_rev_var = (total_var_data[key1] )
        ax.plot(total_rev, label=f'{model}',**style)
        ax.fill_between(range(len(total_avg_data[key1])), 
                        np.float64(total_rev) - (np.sqrt(np.float64(total_rev_var))/scale_factor), 
                        np.float64(total_rev) + (np.sqrt(np.float64(total_rev_var))/scale_factor), 
                        alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    model = 'SIR$^2$'
    style = style_dict[model]
    key1 = f'{info_types[1]}_{model}'
    total_rev = (total_avg_data[key1])/ scale_factor
    total_rev_var = (total_var_data[key1])/ scale_factor
    ax.plot(total_rev,**style)
    ax.fill_between(range(len(total_avg_data[key1])), 
                    np.float64(total_rev) - (np.sqrt(np.float64(total_rev_var))/scale_factor), 
                    np.float64(total_rev) + (np.sqrt(np.float64(total_rev_var))/scale_factor), 
                    alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    if i == 0:
        ax.set_ylabel(r'Total revenue', fontsize=fs)
    ax.set_title(f'$\mu = {mu_A}$', fontsize=fs)  
    ax.set_xlabel(r'Iterations', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data['revenue_SIR$^2$'])+1, 25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
    tick_positions = np.arange(0, subrange+1, 5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    # ax_inset.yaxis.set_major_formatter(formatter)
    ax.text(-0.05, 1.11, f'$\\times 10^{{{power}}}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models))
plt.tight_layout(rect=[0, 0.21, 1, 1])
plt.savefig(f'CournotCompetition/figs/revenue_var.pdf', transparent=True, bbox_inches='tight')
plt.close()
fig, axes = plt.subplots(1, 4, figsize=figuresize)
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    total_var_data = all_mu_A_data[mu_A]['var']
    # ax = fig.add_subplot(gs[0, i])
    ax = axes[i]
    for model in models:
        key1 = f'{info_types[1]}_{model}'
        style = style_dict[model]
        # color = model_colors[model]
        # alpha = company_styles[company]['alpha']
        total_rev = (total_avg_data[key1])/ scale_factor
        total_rev_var = (total_var_data[key1] )
        ax.plot(total_rev, label=f'{model}',**style)
    model = 'SIR$^2$'
    style = style_dict[model]
    key1 = f'{info_types[1]}_{model}'
    total_rev = (total_avg_data[key1])/ scale_factor
    total_rev_var = (total_var_data[key1])/ scale_factor
    ax.plot(total_rev,**style)
    if i == 0:
        ax.set_ylabel(r'Total revenue', fontsize=fs)
    ax.set_title(f'$\mu = {mu_A}$', fontsize=fs)  
    ax.set_xlabel(r'Iterations', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data['revenue_SIR$^2$'])+1, 25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
    tick_positions = np.arange(0, subrange+1, 5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.text(-0.05, 1.11, f'$\\times 10^{{{power}}}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models))
plt.tight_layout(rect=[0, 0.21, 1, 1])
plt.savefig(f'CournotCompetition/figs/revenue.pdf', transparent=True, bbox_inches='tight')
plt.close()

# Plotting quantity
fig, axes = plt.subplots(1, 4, figsize=figuresize)
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    for model in models:    
        key1 = f'{info_types[0]}_{model}'
        all_y_data.extend(total_avg_data[key1])
all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(np.abs(all_y_data)))))
scale_factor = 10 ** power
y_min = min(all_y_data)/ scale_factor -0.2
y_max = max(all_y_data)/ scale_factor +0.2
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    total_var_data = all_mu_A_data[mu_A]['var']
    # ax = fig.add_subplot(gs[0, i])
    ax = axes[i]
    for model in models:
        key1 = f'{info_types[0]}_{model}'
        style = style_dict[model]
        total_quantity = (total_avg_data[key1])/ scale_factor
        total_quantity_var = (total_var_data[key1] )
        ax.plot(total_quantity, label=f'{model}',**style)
        # ax.fill_between(range(len(total_avg_data[key1])), 
        #                 np.float64(total_quantity) - (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
        #                 np.float64(total_quantity) + (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
        #                 alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    model = 'SIR$^2$'
    style = style_dict[model]
    key1 = f'{info_types[0]}_{model}'
    total_quantity = (total_avg_data[key1])/ scale_factor
    total_quantity_var = (total_var_data[key1])/ scale_factor
    ax.plot(total_quantity,**style)
    # ax.fill_between(range(len(total_avg_data[key1])), 
    #             np.float64(total_quantity) - (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
    #             np.float64(total_quantity) + (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
    #             alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    if i == 0:
        ax.set_ylabel(r'Total quantity', fontsize=fs)
    ax.set_title(f'$\mu = {mu_A}$', fontsize=fs)  
    ax.set_xlabel(r'Iterations', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data['quantity_SIR$^2$'])+1, 25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
    tick_positions = np.arange(0, subrange+1, 5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    # ax_inset.yaxis.set_major_formatter(formatter)
    ax.text(-0.05, 1.11, f'$\\times 10^{{{power}}}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models))
plt.tight_layout(rect=[0, 0.21, 1, 1])
plt.savefig(f'CournotCompetition/figs/quantity_var.pdf', transparent=True, bbox_inches='tight')
plt.close()

# Plotting price
fig, axes = plt.subplots(1, 4, figsize=figuresize)
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    for model in models:    
        key1 = f'{info_types[2]}_{model}'
        all_y_data.extend(total_avg_data[key1])
all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(np.abs(all_y_data)))))
scale_factor = 10 ** power
y_min = 0 #min(all_y_data)/ scale_factor -0.2
y_max = max(all_y_data)/ scale_factor +0.2
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    total_var_data = all_mu_A_data[mu_A]['var']
    # ax = fig.add_subplot(gs[0, i])
    ax = axes[i]
    for model in models:
        key1 = f'{info_types[2]}_{model}'
        style = style_dict[model]
        total_price = (total_avg_data[key1])/ scale_factor
        total_price_var = (total_var_data[key1] )
        ax.plot(total_price, label=f'{model}',**style)
        # ax.fill_between(range(len(total_avg_data[key1])), 
        #                 np.float64(total_price) - (np.sqrt(np.float64(total_price_var))/scale_factor), 
        #                 np.float64(total_price) + (np.sqrt(np.float64(total_price_var))/scale_factor), 
        #                 alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    model = 'SIR$^2$'
    style = style_dict[model]
    key1 = f'{info_types[2]}_{model}'
    total_price = (total_avg_data[key1])/ scale_factor
    total_price_var = (total_var_data[key1])/ scale_factor
    ax.plot(total_price,**style)
    # ax.fill_between(range(len(total_avg_data[key1])), 
    #             np.float64(total_price) - (np.sqrt(np.float64(total_price_var))/scale_factor), 
    #             np.float64(total_price) + (np.sqrt(np.float64(total_price_var))/scale_factor), 
    #             alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    if i == 0:
        ax.set_ylabel(r'Price', fontsize=fs)
    ax.set_title(f'$\mu = {mu_A}$', fontsize=fs)  
    ax.set_xlabel(r'Iterations', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data['price_SIR$^2$'])+1, 25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
    tick_positions = np.arange(0, subrange+1, 5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    # ax_inset.yaxis.set_major_formatter(formatter)
    ax.text(-0.05, 1.11, f'$\\times 10^{power}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models))
plt.tight_layout(rect=[0, 0.21, 1, 1])
plt.savefig(f'CournotCompetition/figs/price_var.pdf', transparent=True, bbox_inches='tight')
plt.close()

# Plotting quantity adjuestment
fig, axes = plt.subplots(1, 4, figsize=figuresize)
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    for model in models:
        key1 = f'{info_types[0]}_{model}'
        all_y_data.extend(total_avg_data[key1]-np.sum(data))
all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(np.abs(all_y_data)))))
scale_factor = 10 ** power
y_min = min(all_y_data)/ scale_factor -0.2
y_max = max(all_y_data)/ scale_factor +0.2
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    total_var_data = all_mu_A_data[mu_A]['var']
    # ax = fig.add_subplot(gs[0, i])
    ax = axes[i]
    for model in models:
        key1 = f'{info_types[0]}_{model}'
        style = style_dict[model]
        total_quantity = (total_avg_data[key1]-np.sum(data))/ scale_factor
        total_quantity_var = (total_var_data[key1])
        ax.plot(total_quantity, label=f'{model}',**style)
        # ax.fill_between(range(len(total_avg_data[key1])), 
        #                 np.float64(total_quantity) - (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
        #                 np.float64(total_quantity) + (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
        #                 alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    model = 'SIR$^2$'
    style = style_dict[model]
    key1 = f'{info_types[0]}_{model}'
    total_quantity = (total_avg_data[key1]-np.sum(data))/ scale_factor
    total_quantity_var = (total_var_data[key1])/ scale_factor
    ax.plot(total_quantity,**style)
    # ax.fill_between(range(len(total_avg_data[key1])), 
    #             np.float64(total_quantity) - (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
    #             np.float64(total_quantity) + (np.sqrt(np.float64(total_quantity_var))/scale_factor), 
    #             alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    if i == 0:
        ax.set_ylabel(r'Quantity adjuestment', fontsize=fs-16)
    ax.set_title(f'$\mu = {mu_A}$', fontsize=fs)  
    ax.set_xlabel(r'Iterations', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data['quantity_SIR$^2$'])+1, 25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
    tick_positions = np.arange(0, subrange+1, 5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    # ax_inset.yaxis.set_major_formatter(formatter)
    ax.text(0, 1.11, f'$\\times 10^{{{power}}}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models))
plt.tight_layout(rect=[0, 0.21, 1, 1])
plt.savefig(f'CournotCompetition/figs/quantity_adj_var.pdf', transparent=True, bbox_inches='tight')
plt.close()

# 在绘图部分之后添加时间-性能对比图
# Plotting time-performance comparison
fig, axes = plt.subplots(1, 4, figsize=figuresize)
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    for model in models:    
        key1 = f'{info_types[1]}_{model}'
        all_y_data.extend(total_avg_data[key1])
all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(all_y_data))))
scale_factor = 10 ** power
y_min = 0
y_max = max(all_y_data)/ scale_factor +0.2

for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    ax = axes[i]
    
    for model in models:
        style = style_dict[model]
        
        # 获取收益数据
        revenue_key = f'{info_types[1]}_{model}'
        revenues = total_avg_data[revenue_key] / scale_factor
        
        # 获取时间数据并计算累计时间
        time_key = f'iteration_times_{model}'
        iteration_times = total_avg_data[time_key]
        cumulative_times = np.cumsum(iteration_times)
        
        # 绘制时间-性能曲线
        ax.plot(cumulative_times, revenues, label=f'{model}', **style)
    
    ax.set_title(f'$\mu = {mu_A}$', fontsize=fs)  
    ax.set_xlabel('Running Time', fontsize=fs)
    if i == 0:
        ax.set_ylabel(r'Total revenue', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0,0.03)
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    
    # # 设置x轴格式
    # formatter = ScalarFormatter()
    # formatter.set_scientific(False)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)
    
    ax.text(-0.05, 1.11, f'$\\times 10^{{{power}}}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')

handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models))
plt.tight_layout(rect=[0, 0.21, 1, 1])
plt.savefig(f'CournotCompetition/figs/time_performance_comparison.pdf', transparent=True, bbox_inches='tight')
plt.close()

# 利用 all_mu_A_data 计算总收益和标准差
for mu_A, data in all_mu_A_data.items():
    avg_data = data['avg']
    var_data = data['var']
    for model in models:
        revenue_p1 = avg_data[f'revenue_{model}']
        total_revenue = revenue_p1
        # 取总收益序列的最后一位数字
        final_revenue = total_revenue[-1]
        var_p1 = var_data[f'revenue_{model}']
        total_var = var_p1
        final_var = total_var[-1]
        stat_str = f'{final_revenue:0.0f} $\pm$ {np.sqrt(final_var):0.0f}'
        total_revenue_stats.append({
            'model': model,
            'mu_A': f'$\mu$ = {mu_A}',
            'total_revenue': stat_str
        })

# 创建 DataFrame 并保存为 Excel
df = pd.DataFrame(total_revenue_stats)
pivot_df = df.pivot(index='model', columns='mu_A', values='total_revenue')
# 确保 model 顺序和列表一致
pivot_df = pivot_df.reindex(models)
pivot_df.to_excel('CournotCompetition/figs/total_revenue.xlsx')
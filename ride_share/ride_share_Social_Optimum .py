import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.ticker import EngFormatter
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilsrm import *

## Initialize the game class and set the random seed and initial point
# seed 
seed = 42
np.random.seed(seed)
num_experiments = 10
figuresize=(25, 8)
loc_cap=11
eta=0.001
run_experiment = 0 # 1: run the experiment, 0: load the data
mu_A_list = [0.25,0.50,0.75,1.0] #25,0.50,0.75,1
subrange = 26
gamma = 2.1
BATCH=10
MAXITER=1000
tot_rev=1

# 不同的模型
fs=44
lw=4
lw2 = lw/2
models = ['SIR$^2$', 'RGD','SFB','AGM','OPGD']
# models = ['SIR$^2$', 'OPGD']
companies = ['Lyft', 'Uber']
info_types = ['price','rev', 'demand']
# 定义不同模型对应的颜色
style_dict = {
    'SIR$^2$': {'color': '#FF7F50', 'linestyle': '-', 'linewidth': lw+1},
    'AGM': {'color': '#9467bd', 'linestyle': '--', 'linewidth': lw},
    'RGD': {'color': '#444444', 'linestyle': ':', 'linewidth': lw},
    'SFB': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': lw},
    'OPGD': {'color': '#1f77b4', 'linestyle': (0, (5, 5)), 'linewidth': lw}
}
# 定义不同公司对应的线形和透明度
company_styles = {
    'Lyft': {'linestyle': '-', 'alpha': 1},
    'Uber': {'linestyle': '--', 'alpha': 1}
}

all_mu_A_data = {}
total_revenue_stats = []
all_mu_A_data_path = 'ride_share/data/all_mu_A_data.npy'
if run_experiment:
    for mu_A in mu_A_list:
        print('Runing at mu_A',mu_A)
        mu_AC = mu_A/2
        total_avg_data = {}
        total_var_data = {}
        for p in [0,1,2,3,4]:  #,1,2,3,4
            print('  Runing at price_index',p)
            price_index = p
            data_file_path = f'ride_share/data/mu_A_{mu_A}_{price_index*5+10}_data.npy'
            # set up the game
            loc_lst_index=list(range(0,loc_cap))
            price_lst_index=list(range(0,5))
            price_start = price_index*5+10
            
            all_data={}
            for num_exper in range(num_experiments):
                seed = seed+1
                np.random.seed(seed)
                params = {}
                params['A1']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
                params['A2']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
                params['Ac1'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
                params['Ac2'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
                print('    Runing at number',num_exper+1,'trail')
                x0=np.random.rand(2,loc_cap)*5 # initial point in the range [0,5]
                all_data[num_exper]={}
                all_data[num_exper]['x0']=x0
                ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=False, params=params,maxx=10)
                ddgame.setup_distribution()

                # run all cases
                dic_data = []
                dic_data.append(ddgame.runRR(gamma = gamma,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runRGD(x0,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runSFB(x0,price_index=price_index,eta=eta,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runAGD(x0,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runOPGD(x0,price_index=price_index,eta=eta,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))

                for model, dic in zip(models, dic_data):
                    # 从字典中获取 x 数据
                    x = np.asarray(dic['x'])
                    for i, company in enumerate(companies):
                        # 计算价格
                        price = np.mean(x[:, i, :], axis=1) + price_start
                        # 获取收入
                        rev_key = f'revenue_total_p{i + 1}'
                        rev = np.asarray(dic[rev_key])
                        # 计算需求
                        demand_key = f'demand_p{i + 1}'
                        demand = np.asarray(dic[demand_key])

                        all_data[num_exper][f'{info_types[0]}_{model}_{company}'] = price
                        all_data[num_exper][f'{info_types[1]}_{model}_{company}'] = rev
                        all_data[num_exper][f'{info_types[2]}_{model}_{company}'] = demand

            avg_data = {}
            var_data = {}  # 新增一个字典来存储方差
            for model in models:
                for company in companies:
                    for info_type in info_types:
                        key = f'{info_type}_{model}_{company}'
                        all_values = [all_data[num_exper][key] for num_exper in range(num_experiments)]
                        all_values_arr = np.asarray(all_values)
                        avg_data[key] = np.mean(all_values_arr, axis=0)
                        var_data[key] = np.var(all_values_arr, axis=0)  # 计算方差

            for key, value in avg_data.items():
                if key in total_avg_data:
                    total_avg_data[key] += value
                else:
                    total_avg_data[key] = value

            for key, value in var_data.items():
                if key in total_var_data:
                    total_var_data[key] += value
                else:
                    total_var_data[key] = value

        all_mu_A_data[mu_A] = {'avg': total_avg_data, 'var': total_var_data}
    np.save(all_mu_A_data_path, all_mu_A_data)
else:
    all_mu_A_data = np.load(all_mu_A_data_path, allow_pickle=True).item()

## Plotting variance
info_type = 'total_revenue'
fig, axes = plt.subplots(1, 4, figsize=figuresize)
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    for model in models:    
        key1 = f'{info_types[1]}_{model}_{companies[0]}'
        key2 = f'{info_types[1]}_{model}_{companies[1]}'
        all_y_data.extend(total_avg_data[key1] + total_avg_data[key2])
all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(np.abs(all_y_data)))))
scale_factor = 10 ** power

y_min = min(all_y_data)/ scale_factor
y_max = max(all_y_data)/ scale_factor +0.1
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = all_mu_A_data[mu_A]['avg']
    total_var_data = all_mu_A_data[mu_A]['var']
    # ax = fig.add_subplot(gs[0, i])
    ax = axes[i]
    for model in models:
        key1 = f'{info_types[1]}_{model}_{companies[0]}'
        key2 = f'{info_types[1]}_{model}_{companies[1]}'
        style = style_dict[model]
        # color = model_colors[model]
        # alpha = company_styles[company]['alpha']
        total_rev = (total_avg_data[key1] + total_avg_data[key2])/ scale_factor
        total_rev_var = (total_var_data[key1] + total_var_data[key2])
        ax.plot(total_rev, label=f'{model}',**style)
        ax.fill_between(range(len(total_avg_data[key1])), total_rev - (np.sqrt(total_rev_var)/scale_factor), total_rev + (np.sqrt(total_rev_var)/scale_factor), alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
        # ax_inset.plot(total_rev[:subrange], color=color, alpha=alpha, lw=lw)
    model = 'SIR$^2$'
    style = style_dict[model]
    key1 = f'{info_types[1]}_{model}_{companies[0]}'
    key2 = f'{info_types[1]}_{model}_{companies[1]}'
    total_rev = (total_avg_data[key1] + total_avg_data[key2])/ scale_factor
    total_rev_var = (total_var_data[key1] + total_var_data[key2])/ scale_factor
    ax.plot(total_rev,**style)
    ax.fill_between(range(len(total_rev)), total_rev - (np.sqrt(total_rev_var)/scale_factor), total_rev + (np.sqrt(total_rev_var)/scale_factor), alpha=0.2, color=style_dict[model]['color'],edgecolor='none')
    if i == 0:
        ax.set_ylabel(r'Total revenue', fontsize=fs)
    ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)  
    ax.set_xlabel(r'Iterations', fontsize=fs)
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.5)
    ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data['rev_SIR$^2$_Lyft'])+1, 250)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
    tick_positions = np.arange(0, subrange+1, 5)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    # ax_inset.yaxis.set_major_formatter(formatter)
    ax.text(-0.1, 1.1, f'$\\times 10^{power}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models) * len(companies))
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig(f'ride_share/figs/{info_type}_var.pdf', transparent=True, bbox_inches='tight')
plt.close()


info_types_extended = ['total_price', 'demand', 'each_revenue', 'total_revenue']
for info_type in info_types_extended:
    if info_type != 'total_revenue':
        new_figsize = (figuresize[0], figuresize[1]+5)
        fig, axes = plt.subplots(2, 4, figsize=new_figsize)
    for company in companies:
        if info_type == 'total_revenue':
            fig, axes = plt.subplots(1, 4, figsize=figuresize)
        # axes = axes.flatten()
        # 找出所有子图数据在该大图对应的指标下的最小值和最大值
        all_y_data = []
        for i, mu_A in enumerate(mu_A_list):
            total_avg_data = all_mu_A_data[mu_A]['avg']
            if info_type == 'total_price':
                for model in models:
                    key = f'{info_types[0]}_{model}_{company}'
                    all_y_data.extend(total_avg_data[key] / 5)
            elif info_type == 'demand':
                for model in models:
                    key = f'{info_types[2]}_{model}_{company}'
                    all_y_data.extend(np.sum(total_avg_data[key], axis=1))
            elif info_type == 'each_revenue':
                mean_val = 1
                for model in models:
                    key = f'{info_types[1]}_{model}_{company}'
                    all_y_data.extend(running_mean(total_avg_data[key], N=mean_val))
            elif info_type == 'total_revenue':                 
                for model in models:    
                    key1 = f'{info_types[1]}_{model}_{companies[0]}'
                    key2 = f'{info_types[1]}_{model}_{companies[1]}'
                    all_y_data.extend(total_avg_data[key1] + total_avg_data[key2])

        all_y_data = np.array(all_y_data)
        if np.allclose(all_y_data, 0):
            power = 0
        else:
            power = int(np.floor(np.log10(np.max(np.abs(all_y_data)))))
        scale_factor = 10 ** power

        y_min = min(all_y_data)/ scale_factor
        y_max = max(all_y_data)/ scale_factor +0.1

        for i, mu_A in enumerate(mu_A_list):
            total_avg_data = all_mu_A_data[mu_A]['avg']
            total_var_data = all_mu_A_data[mu_A]['var']
            # ax = fig.add_subplot(gs[0, i])
            # ax_inset = fig.add_subplot(gs[1, i])

            if info_type == 'total_price':
                #ax = axes[i]
                if company == 'Lyft':
                    ax = axes[0,i]
                else:
                    ax = axes[1,i]
                for model in models:
                    key = f'{info_types[0]}_{model}_{company}'
                    style = style_dict[model]
                    ax.plot(total_avg_data[key] / (5 * scale_factor), label=f'{model}',**style)
                    # ax_inset.plot(total_avg_data[key][:subrange]/ (5 * scale_factor), color=color, linestyle=linestyle, alpha=alpha, lw=lw2)
                style = style_dict['SIR$^2$']
                ax.plot(total_avg_data[f'price_SIR$^2$_{company}'] / (5 * scale_factor), **style)
                if i == 0 :
                    ax.set_ylabel(f'{company} prices', fontsize=fs)
                if company == 'Lyft':
                    ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)  
                else:
                    ax.set_xlabel(r'Iterations', fontsize=fs)
                
            elif info_type == 'demand':
                #ax = axes[i]
                if company == 'Lyft':
                    ax = axes[0,i]
                else:
                    ax = axes[1,i]
                for model in models:
                    key = f'{info_types[2]}_{model}_{company}'
                    style = style_dict[model]
                    # color = model_colors[model]
                    # linestyle = company_styles[company]['linestyle']
                    # alpha = company_styles[company]['alpha']
                    ax.plot(np.sum(total_avg_data[key]/ scale_factor, axis=1), label=f'{model}',**style)
                    # ax_inset.plot(np.sum(total_avg_data[key][:subrange]/ scale_factor, axis=1), color=color, linestyle=linestyle, alpha=alpha, lw=lw2)
                style = style_dict['SIR$^2$']
                ax.plot(np.sum(total_avg_data[f'demand_SIR$^2$_{company}'], axis=1), **style)
                if i == 0 :
                    ax.set_ylabel(f'{company} demand', fontsize=fs)
                if company == 'Lyft':
                    ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)  
                else:
                    ax.set_xlabel(r'Iterations', fontsize=fs)
                
            elif info_type == 'each_revenue':
                #ax = axes[i]
                if company == 'Lyft':
                    ax = axes[0,i]
                else:
                    ax = axes[1,i]
                mean_val = 1
                for model in models:
                    key = f'{info_types[1]}_{model}_{company}'
                    style = style_dict[model]
                    # color = model_colors[model]
                    # linestyle = company_styles[company]['linestyle']
                    # alpha = company_styles[company]['alpha']
                    revenue_data = running_mean(total_avg_data[key]/ scale_factor, N=mean_val)
                    ax.plot(revenue_data, label=f'{model}',**style)
                    # ax_inset.plot(revenue_data[:subrange], color=color, linestyle=linestyle, alpha=alpha, lw=lw2)
                    # ax.fill_between(range(len(total_avg_data[key])), running_mean(total_avg_data[key], N=mean_val) - np.sqrt(
                    #     total_var_data[key]), running_mean(total_avg_data[key], N=mean_val) + np.sqrt(total_var_data[key]),color=color, alpha=0.5)
                style = style_dict['SIR$^2$']
                ax.plot(running_mean(total_avg_data[f'rev_SIR$^2$_{company}'], N=mean_val),**style)
                if i == 0:
                    ax.set_ylabel(f'{company} revenue', fontsize=fs)
                if company == 'Lyft':
                    ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)  
                else:
                    ax.set_xlabel(r'Iterations', fontsize=fs)

            elif info_type == 'total_revenue':
                ax = axes[i]
                for model in models:
                    key1 = f'{info_types[1]}_{model}_{companies[0]}'
                    key2 = f'{info_types[1]}_{model}_{companies[1]}'
                    style = style_dict[model]
                    # color = model_colors[model]
                    # alpha = company_styles[company]['alpha']
                    total_rev = (total_avg_data[key1] + total_avg_data[key2])/ scale_factor
                    ax.plot(total_rev, label=f'{model}',**style)
                    # ax_inset.plot(total_rev[:subrange], color=color, alpha=alpha, lw=lw)
                style = style_dict['SIR$^2$']
                ax.plot((total_avg_data['rev_SIR$^2$_Lyft'] + total_avg_data['rev_SIR$^2$_Uber'])/ scale_factor,**style)
                if i == 0:
                    ax.set_ylabel(r'Total revenue', fontsize=fs)
                ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)  
                ax.set_xlabel(r'Iterations', fontsize=fs)
            ax.grid(True)
            ax.tick_params(labelsize=fs*0.5)
            ax.set_ylim(y_min, y_max)
            # ax_inset.set_ylim(y_min, y_max)
            # ax_inset.set_xlim(-1, subrange+0.5)
            # ax_inset.grid(True)
            tick_positions = np.arange(0, len(total_avg_data['rev_SIR$^2$_Lyft'])+1, 250)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, fontsize=fs*0.7)
            tick_positions = np.arange(0, subrange+1, 5)
            # ax_inset.set_xticks(tick_positions)
            # ax_inset.set_xticklabels(tick_positions, fontsize=fs*0.7)
            # ax_inset.tick_params(labelsize=fs*0.7)

            # # 添加虚线方框
            # rect = Rectangle((0, y_min), subrange, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
            # ax.add_patch(rect)
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            # ax_inset.yaxis.set_major_formatter(formatter)
            ax.text(-0.1, 1.1, f'$\\times 10^{power}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')
            # ax_inset.text(-0.1, 1.1, f'$\\times 10^{power}$', transform=ax_inset.transAxes, fontsize=fs*0.7, verticalalignment='top')
        if company =='Uber':
            handles, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models) * len(companies))
            # plt.tight_layout(rect=[0, 0.18, 1, 1])
            if info_type == 'total_revenue':
                # fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=len(models) * len(companies))
                plt.tight_layout(rect=[0, 0.18, 1, 1])
            else:
                # fig.legend(handles, labels, loc='lower center', fontsize=fs*0.82, ncol=len(models) * len(companies)//2)
                plt.tight_layout(rect=[0, 0.12, 1, 1])
            plt.savefig(f'ride_share/figs/{info_type}.pdf', transparent=True, bbox_inches='tight')
            plt.close()

# 利用 all_mu_A_data 计算总收益和标准差
for mu_A, data in all_mu_A_data.items():
    avg_data = data['avg']
    var_data = data['var']
    for model in models:
        revenue_p1 = avg_data[f'rev_{model}_Lyft']
        revenue_p2 = avg_data[f'rev_{model}_Uber']
        total_revenue = revenue_p1 + revenue_p2
        # 取总收益序列的最后一位数字
        final_revenue = total_revenue[-1]
        var_p1 = var_data[f'rev_{model}_Lyft']
        var_p2 = var_data[f'rev_{model}_Uber']
        total_var = var_p1 + var_p2
        final_var = total_var[-1]
        stat_str = f'{final_revenue:0.0f} $\pm$ {np.sqrt(final_var):0.0f}'
        total_revenue_stats.append({
            'model': model,
            'mu_A': f'$\mu_A$ = {mu_A}',
            'total_revenue': stat_str
        })

# 创建 DataFrame 并保存为 Excel
df = pd.DataFrame(total_revenue_stats)
pivot_df = df.pivot(index='model', columns='mu_A', values='total_revenue')
# 确保 model 顺序和列表一致
pivot_df = pivot_df.reindex(models)
pivot_df.to_excel('ride_share/figs/total_revenue.xlsx')
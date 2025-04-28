import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilsrm import *

## Initialize the game class and set the random seed and initial point
# seed 
np.random.seed(42)
num_experiments = 10
figuresize=(14, 7)
loc_cap=11
eta=0.001 
run_experiment = 1 # 1: run the experiment, 0: load the data
mu_A_list = [0.25,0.50,0.75,1.0] #25,0.50,0.75,1

gamma = 2.1
BATCH=10
MAXITER=100
tot_rev=1

# 不同的模型
models = ['RR', 'RGD','SFB','AGM','OPGD']
companies = ['Lyft', 'Uber']
info_types = ['price','rev', 'demand']
# 定义不同模型对应的颜色
model_colors = {
    'RR': '#FF7F50',
    'AGM': '#9467bd',
    'RGD': '#444444',
    'SFB': '#2ca02c',
    'OPGD': '#1f77b4',
}
# 定义不同公司对应的线形和透明度
company_styles = {
    'Lyft': {'linestyle': '-', 'alpha': 1},
    'Uber': {'linestyle': '--', 'alpha': 1}
}
fs=24
lw=2
all_mu_A_data = {}
total_revenue_stats = []
for mu_A in mu_A_list:
    mu_AC = mu_A/2
    params = {}
    params['A1']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
    params['A2']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
    params['Ac1'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
    params['Ac2'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)

    total_avg_data = {}
    total_var_data = {}
    for p in [0,1,2,3,4]:  #,1,2,3,4
        price_index = p
        data_file_path = f'ride_share/data/mu_A_{mu_A}_{price_index*5+10}_data.npy'
        # set up the game
        loc_lst_index=list(range(0,loc_cap))
        price_lst_index=list(range(0,5))
        price_start = price_index*5+10
        if run_experiment:
            all_data={}
            for num_exper in range(num_experiments):
                print('Runing at number',num_exper+1,'trail')
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

            combined_data = {'avg': avg_data, 'var': var_data}  # 合并均值和方差数据
            np.save(data_file_path, combined_data)
        else:
            combined_data = np.load(data_file_path, allow_pickle=True).item()
            avg_data = combined_data['avg']
            var_data = combined_data['var']

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
  
        # fname='ride_share/figs_'+str(mu_A)+'/'+str((price_index*5+10))+'_prices.'
        # plt.figure(figsize=figuresize)

        # for model in models:
        #     for company in companies:
        #         key = f'{info_types[0]}_{model}_{company}'
        #         color = model_colors[model]
        #         linestyle = company_styles[company]['linestyle']
        #         alpha = company_styles[company]['alpha']
        #         plt.plot(avg_data[key], label=f'{model}, {company}',lw=lw,color=color, linestyle=linestyle, alpha=alpha)
        # plt.plot(avg_data['price_RR_Lyft'], lw=lw,color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
        # plt.plot(avg_data['price_RR_Uber'], lw=lw,color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
        # plt.grid(True)
        # plt.xlabel(r'iterations', fontsize=fs)
        # plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
        # plt.tick_params(labelsize=fs-2)
        # plt.ylabel(r'prices', fontsize=fs)
        # plt.tight_layout()
        # plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

        # fname='ride_share/figs_'+str(mu_A)+'/'+str((price_index*5+10))+'_demand.'
        # plt.figure(figsize=figuresize)

        # for model in models:
        #     for company in companies:
        #         key = f'{info_types[2]}_{model}_{company}'
        #         color = model_colors[model]
        #         linestyle = company_styles[company]['linestyle']
        #         alpha = company_styles[company]['alpha']
        #         plt.plot(np.sum(avg_data[key], axis=1), label=f'{model}, {company}',lw=lw,color=color, linestyle=linestyle, alpha=alpha)
        # plt.plot(np.sum(avg_data['demand_RR_Lyft'], axis=1),lw=lw, color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
        # plt.plot(np.sum(avg_data['demand_RR_Uber'], axis=1),lw=lw, color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
        # plt.grid(True)
        # plt.xlabel(r'iterations', fontsize=fs)
        # plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
        # plt.tick_params(labelsize=fs-2)
        # plt.ylabel(r'demand', fontsize=fs)
        # plt.tight_layout()
        # plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

        # fname='ride_share/figs_'+str(mu_A)+'/'+str((price_index*5+10))+'_each_revenue.'
        # plt.figure(figsize=figuresize)
        # mean_val=20
        # for model in models:
        #     for company in companies:
        #         key = f'{info_types[1]}_{model}_{company}'
        #         color = model_colors[model]
        #         linestyle = company_styles[company]['linestyle']
        #         alpha = company_styles[company]['alpha']
        #         plt.plot(running_mean(avg_data[key],N=mean_val), label=f'{model}, {company}',lw=lw,
        #                         color=color, linestyle=linestyle, alpha=alpha)
        # plt.plot(running_mean(avg_data['rev_RR_Lyft'],N=mean_val),lw=lw,color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
        # plt.plot(running_mean(avg_data['rev_RR_Uber'],N=mean_val),lw=lw,color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
        # plt.grid(True)
        # plt.xlabel(r'iterations', fontsize=fs)
        # plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
        # plt.tick_params(labelsize=fs-2)
        # plt.ylabel(r'revenue', fontsize=fs)
        # plt.tight_layout()
        # plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

        # fname='ride_share/figs_'+str(mu_A)+'/'+str((price_index*5+10))+'_total_revenue.'
        # # new_figuresize = (figuresize[0] + 5,) + figuresize[1:]
        # plt.figure(figsize=figuresize)
        # for model in models:
        #     key1 = f'{info_types[1]}_{model}_{companies[0]}'
        #     key2 = f'{info_types[1]}_{model}_{companies[1]}'
        #     color = model_colors[model]
        #     alpha = company_styles[company]['alpha']
        #     plt.plot(avg_data[key1]+avg_data[key2], label=f'{model}',lw=lw,color=color,alpha=alpha)
        # plt.plot(avg_data['rev_RR_Lyft']+avg_data['rev_RR_Uber'],lw=lw,color=model_colors['RR'],alpha=alpha)
        # plt.grid(True)
        # plt.xlabel(r'iterations', fontsize=fs+2)
        # plt.tick_params(labelsize=fs-2)
        # plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5)) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
        # plt.ylabel(r'total revenue', fontsize=fs+2)
        # plt.tight_layout()
        # plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')
    
    fname='ride_share/figs/result_'+str(mu_A)+'/00_total_prices.'
    plt.figure(figsize=figuresize)

    for model in models:
        for company in companies:
            key = f'{info_types[0]}_{model}_{company}'
            color = model_colors[model]
            linestyle = company_styles[company]['linestyle']
            alpha = company_styles[company]['alpha']
            plt.plot(total_data[key]/5, label=f'{model}, {company}',lw=lw,color=color, linestyle=linestyle, alpha=alpha)
    plt.plot(total_data['price_RR_Lyft']/5, lw=lw,color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
    plt.plot(total_data['price_RR_Uber']/5, lw=lw,color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
    plt.grid(True)
    plt.xlabel(r'iterations', fontsize=fs)
    plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
    plt.tick_params(labelsize=fs-2)
    plt.ylabel(r'prices', fontsize=fs)
    plt.tight_layout()
    plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

    fname='ride_share/figs/result_'+str(mu_A)+'/00_total_demand.'
    plt.figure(figsize=figuresize)

    for model in models:
        for company in companies:
            key = f'{info_types[2]}_{model}_{company}'
            color = model_colors[model]
            linestyle = company_styles[company]['linestyle']
            alpha = company_styles[company]['alpha']
            plt.plot(np.sum(total_data[key], axis=1), label=f'{model}, {company}',lw=lw,color=color, linestyle=linestyle, alpha=alpha)
    plt.plot(np.sum(total_data['demand_RR_Lyft'], axis=1),lw=lw, color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
    plt.plot(np.sum(total_data['demand_RR_Uber'], axis=1),lw=lw, color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
    plt.grid(True)
    plt.xlabel(r'iterations', fontsize=fs)
    plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
    plt.tick_params(labelsize=fs-2)
    plt.ylabel(r'demand', fontsize=fs)
    plt.tight_layout()
    plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

    fname='ride_share/figs/result_'+str(mu_A)+'/00_total_each_revenue.'
    plt.figure(figsize=figuresize)
    mean_val=20
    for model in models:
        for company in companies:
            key = f'{info_types[1]}_{model}_{company}'
            color = model_colors[model]
            linestyle = company_styles[company]['linestyle']
            alpha = company_styles[company]['alpha']
            plt.plot(running_mean(total_data[key],N=mean_val), label=f'{model}, {company}',lw=lw,
                            color=color, linestyle=linestyle, alpha=alpha)
    plt.plot(running_mean(total_data['rev_RR_Lyft'],N=mean_val),lw=lw,color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
    plt.plot(running_mean(total_data['rev_RR_Uber'],N=mean_val),lw=lw,color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
    plt.grid(True)
    plt.xlabel(r'iterations', fontsize=fs)
    plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
    plt.tick_params(labelsize=fs-2)
    plt.ylabel(r'revenue', fontsize=fs)
    plt.tight_layout()
    plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

    fname='ride_share/figs/result_'+str(mu_A)+'/00_total_total_revenue.'
    # new_figuresize = (figuresize[0] - 2,) + figuresize[1:]
    plt.figure(figsize=figuresize)
    for model in models:
        key1 = f'{info_types[1]}_{model}_{companies[0]}'
        key2 = f'{info_types[1]}_{model}_{companies[1]}'
        color = model_colors[model]
        alpha = company_styles[company]['alpha']
        plt.plot(total_data[key1]+total_data[key2], label=f'{model}',lw=lw,color=color,alpha=alpha)
    plt.plot(total_data['rev_RR_Lyft']+total_data['rev_RR_Uber'],lw=lw,color=model_colors['RR'],alpha=alpha)
    plt.grid(True)
    plt.xlabel(r'iterations', fontsize=fs+2)
    plt.tick_params(labelsize=fs-2)
    plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5)) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
    plt.ylabel(r'total revenue', fontsize=fs+2)
    plt.tight_layout()
    plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

# 绘制 4 个大图
info_types_extended = ['total_price', 'demand', 'each_revenue', 'total_revenue']
for info_type in info_types_extended:
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    axes = axes.flatten()

    # 找出所有子图数据在该大图对应的指标下的最小值和最大值
    all_y_data = []
    for i, mu_A in enumerate(mu_A_list):
        total_data = all_mu_A_data[mu_A]
        if info_type == 'total_price':
            for model in models:
                for company in companies:
                    key = f'{info_types[0]}_{model}_{company}'
                    all_y_data.extend(total_data[key] / 5)
        elif info_type == 'demand':
            for model in models:
                for company in companies:
                    key = f'{info_types[2]}_{model}_{company}'
                    all_y_data.extend(np.sum(total_data[key], axis=1))
        elif info_type == 'each_revenue':
            mean_val = 1
            for model in models:
                for company in companies:
                    key = f'{info_types[1]}_{model}_{company}'
                    all_y_data.extend(running_mean(total_data[key], N=mean_val))
        elif info_type == 'total_revenue':
            for model in models:
                key1 = f'{info_types[1]}_{model}_{companies[0]}'
                key2 = f'{info_types[1]}_{model}_{companies[1]}'
                all_y_data.extend(total_data[key1] + total_data[key2])

    y_min = min(all_y_data)
    y_max = max(all_y_data)

    for i, mu_A in enumerate(mu_A_list):
        total_data = all_mu_A_data[mu_A]
        ax = axes[i]

        if info_type == 'total_price':
            for model in models:
                for company in companies:
                    key = f'{info_types[0]}_{model}_{company}'
                    color = model_colors[model]
                    linestyle = company_styles[company]['linestyle']
                    alpha = company_styles[company]['alpha']
                    ax.plot(total_data[key] / 5, label=f'{model}, {company}', lw=lw, color=color, linestyle=linestyle,
                            alpha=alpha)
            ax.plot(total_data['price_RR_Lyft'] / 5, lw=lw, color=model_colors['RR'],
                    linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
            ax.plot(total_data['price_RR_Uber'] / 5, lw=lw, color=model_colors['RR'],
                    linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
            if i == 0:
                ax.set_ylabel(r'prices', fontsize=fs)
            ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)
            ax.grid(True)
            ax.tick_params(labelsize=fs-2)
            ax.set_xlabel(r'iterations', fontsize=fs)
            
        elif info_type == 'demand':
            for model in models:
                for company in companies:
                    key = f'{info_types[2]}_{model}_{company}'
                    color = model_colors[model]
                    linestyle = company_styles[company]['linestyle']
                    alpha = company_styles[company]['alpha']
                    ax.plot(np.sum(total_data[key], axis=1), label=f'{model}, {company}', lw=lw, color=color,
                            linestyle=linestyle, alpha=alpha)
            ax.plot(np.sum(total_data['demand_RR_Lyft'], axis=1), lw=lw, color=model_colors['RR'],
                    linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
            ax.plot(np.sum(total_data['demand_RR_Uber'], axis=1), lw=lw, color=model_colors['RR'],
                    linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
            if i == 0:
                ax.set_ylabel(r'demand', fontsize=fs)
            ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)
            ax.grid(True)
            ax.tick_params(labelsize=fs-2)
            ax.set_xlabel(r'iterations', fontsize=fs)
            
        elif info_type == 'each_revenue':
            mean_val = 1
            for model in models:
                for company in companies:
                    key = f'{info_types[1]}_{model}_{company}'
                    color = model_colors[model]
                    linestyle = company_styles[company]['linestyle']
                    alpha = company_styles[company]['alpha']
                    ax.plot(running_mean(total_data[key], N=mean_val), label=f'{model}, {company}', lw=lw,
                            color=color, linestyle=linestyle, alpha=alpha)
            ax.plot(running_mean(total_data['rev_RR_Lyft'], N=mean_val), lw=lw, color=model_colors['RR'],
                    linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
            ax.plot(running_mean(total_data['rev_RR_Uber'], N=mean_val), lw=lw, color=model_colors['RR'],
                    linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
            if i == 0:
                ax.set_ylabel(r'revenue', fontsize=fs)
            ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs)
            ax.grid(True)
            ax.set_xlabel(r'iterations', fontsize=fs)
            ax.tick_params(labelsize=fs-2)

        elif info_type == 'total_revenue':
            for model in models:
                key1 = f'{info_types[1]}_{model}_{companies[0]}'
                key2 = f'{info_types[1]}_{model}_{companies[1]}'
                color = model_colors[model]
                alpha = company_styles[company]['alpha']
                ax.plot(total_data[key1] + total_data[key2], label=f'{model}', lw=lw, color=color, alpha=alpha)
            ax.plot(total_data['rev_RR_Lyft'] + total_data['rev_RR_Uber'], lw=lw, color=model_colors['RR'], alpha=alpha)
            if i == 0:
                ax.set_ylabel(r'total revenue', fontsize=fs)
            ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs) 
            ax.grid(True)
            ax.set_xlabel(r'iterations', fontsize=fs)
            ax.tick_params(labelsize=fs-2)
        ax.set_ylim(y_min, y_max)
    handles, labels = axes[0].get_legend_handles_labels()
    if info_type == 'total_revenue':
        fig.legend(handles, labels, loc='lower center', fontsize=fs - 2, ncol=len(models) * len(companies))
        plt.tight_layout(rect=[0, 0.1, 1, 1])
    else:
        fig.legend(handles, labels, loc='lower center', fontsize=fs - 2, ncol=len(models) * len(companies)/2)
        plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(f'ride_share/figs/{info_type}.pdf', transparent=True, bbox_inches='tight')
    plt.close()

# 利用 all_mu_A_data 计算总收益和标准差
for mu_A, data in all_mu_A_data.items():
    for model in models:
        all_last_digits = []
        for exp in range(num_experiments):
            revenue_p1 = data[f'rev_{model}_Lyft']
            revenue_p2 = data[f'rev_{model}_Uber']
            total_revenue = revenue_p1 + revenue_p2
            # 取总收益序列的最后一位数字
            last_digit = total_revenue[-1]
            all_last_digits.append(last_digit)

        all_last_digits = np.array(all_last_digits)
        mean_last_digit = np.mean(all_last_digits)
        std_last_digit = np.std(all_last_digits)
        # 使用 {:0.0f} 保留整数部分
        # stat_str = f'{mean_last_digit:0.0f} $\pm$ {std_last_digit:0.2f}'
        stat_str = f'{mean_last_digit:0.0f}'
        total_revenue_stats.append({
            'model': model,
            'mu_A': f'$\mu_A$={mu_A}',
            'total_revenue': stat_str
        })

# 创建 DataFrame 并保存为 Excel
df = pd.DataFrame(total_revenue_stats)
pivot_df = df.pivot(index='model', columns='mu_A', values='total_revenue')
# 确保 model 顺序和列表一致
pivot_df = pivot_df.reindex(models)
pivot_df.to_excel('ride_share/figs/total_revenue.xlsx')

import numpy as np
import matplotlib.pyplot as plt
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilsrm import *

## Initialize the game class and set the random seed and initial point
# seed 
np.random.seed(42)
num_experiments = 2
figuresize=(14, 7)
loc_cap=11
eta=0.001 
run_experiment = 1 # 1: run the experiment, 0: load the data
mu_A_list = [0.5] #25,0.50,0.75,1

gamma = 2.1
BATCH=10
MAXITER=1000
tot_rev=1

# 不同的模型
models = ['RR','AGM', 'RGD','SFB','OPGD']
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
lw=4
for mu_A in mu_A_list:
    mu_AC = mu_A/2
    params = {}
    params['A1']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
    params['A2']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
    params['Ac1'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
    params['Ac2'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
    for p in [0,1,2,3,4]:  #,1,2,3,4
        price_index = p
        data_file_path = f'ride_share/data/{price_index*5+10}_data.npy'
        # set up the game
        loc_lst_index=list(range(0,loc_cap))
        price_lst_index=list(range(0,5))
        price_start = price_index*5+10
        if run_experiment:
            all_data={}
            for num_exper in range(num_experiments):
                print('Runing at number',num_exper+1,'trail')
                x0=np.random.rand(2,loc_cap)
                all_data[num_exper]={}
                all_data[num_exper]['x0']=x0
                ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=False, params=params,maxx=10)
                ddgame.setup_distribution()

                # run all cases
                dic_data = []
                dic_data.append(ddgame.runRR(gamma = gamma,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runAGD(x0,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runRGD(x0,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
                dic_data.append(ddgame.runSFB(x0,price_index=price_index,eta=eta,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev))
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
            for model in models:
                for company in companies:
                    for info_type in info_types:
                        key = f'{info_type}_{model}_{company}'
                        all_values = [all_data[num_exper][key] for num_exper in range(num_experiments)]
                        avg_data[key] = np.mean(np.asarray(all_values), axis=0)
            np.save(data_file_path, avg_data)
        else:
            avg_data = np.load(data_file_path, allow_pickle=True).item()


        fname='ride_share/figs/'+str((price_index*5+10))+'_prices.'
        plt.figure(figsize=figuresize)

        for model in models:
            for company in companies:
                key = f'{info_types[0]}_{model}_{company}'
                color = model_colors[model]
                linestyle = company_styles[company]['linestyle']
                alpha = company_styles[company]['alpha']
                plt.plot(avg_data[key], label=f'{model}, {company}',lw=lw,color=color, linestyle=linestyle, alpha=alpha)
        plt.plot(avg_data['price_RR_Lyft'], lw=lw,color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
        plt.plot(avg_data['price_RR_Uber'], lw=lw,color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
        plt.grid(True)
        plt.xlabel(r'iterations', fontsize=fs)
        plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
        plt.tick_params(labelsize=fs-2)
        plt.ylabel(r'prices', fontsize=fs)
        plt.tight_layout()
        plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

        fname='ride_share/figs/'+str((price_index*5+10))+'_demand.'
        plt.figure(figsize=figuresize)

        for model in models:
            for company in companies:
                key = f'{info_types[2]}_{model}_{company}'
                color = model_colors[model]
                linestyle = company_styles[company]['linestyle']
                alpha = company_styles[company]['alpha']
                plt.plot(np.sum(avg_data[key], axis=1), label=f'{model}, {company}',lw=lw,color=color, linestyle=linestyle, alpha=alpha)
        plt.plot(np.sum(avg_data['demand_RR_Lyft'], axis=1),lw=lw, color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
        plt.plot(np.sum(avg_data['demand_RR_Uber'], axis=1),lw=lw, color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
        plt.grid(True)
        plt.xlabel(r'iterations', fontsize=fs)
        plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
        plt.tick_params(labelsize=fs-2)
        plt.ylabel(r'demand', fontsize=fs)
        plt.tight_layout()
        plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

        fname='ride_share/figs/'+str((price_index*5+10))+'_each_revenue.'
        plt.figure(figsize=figuresize)
        mean_val=20
        for model in models:
            for company in companies:
                key = f'{info_types[1]}_{model}_{company}'
                color = model_colors[model]
                linestyle = company_styles[company]['linestyle']
                alpha = company_styles[company]['alpha']
                plt.plot(running_mean(avg_data[key],N=mean_val), label=f'{model}, {company}',lw=lw,
                                color=color, linestyle=linestyle, alpha=alpha)
        plt.plot(running_mean(avg_data['rev_RR_Lyft'],N=mean_val),lw=lw,color=model_colors['RR'], linestyle=company_styles['Lyft']['linestyle'], alpha=alpha)
        plt.plot(running_mean(avg_data['rev_RR_Uber'],N=mean_val),lw=lw,color=model_colors['RR'], linestyle=company_styles['Uber']['linestyle'], alpha=alpha)
        plt.grid(True)
        plt.xlabel(r'iterations', fontsize=fs)
        plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5))
        plt.tick_params(labelsize=fs-2)
        plt.ylabel(r'revenue', fontsize=fs)
        plt.tight_layout()
        plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

        fname='ride_share/figs/'+str((price_index*5+10))+'_total_revenue.'
        # new_figuresize = (figuresize[0] + 5,) + figuresize[1:]
        plt.figure(figsize=figuresize)
        for model in models:
            key1 = f'{info_types[1]}_{model}_{companies[0]}'
            key2 = f'{info_types[1]}_{model}_{companies[1]}'
            color = model_colors[model]
            alpha = company_styles[company]['alpha']
            plt.plot(avg_data[key1]+avg_data[key2], label=f'{model}',lw=lw,color=color,alpha=alpha)
        plt.plot(avg_data['rev_RR_Lyft']+avg_data['rev_RR_Uber'],lw=lw,color=model_colors['RR'],alpha=alpha)
        plt.grid(True)
        plt.xlabel(r'iterations', fontsize=fs+2)
        plt.tick_params(labelsize=fs-2)
        plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.45,0.5)) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
        plt.ylabel(r'total revenue', fontsize=fs+2)
        plt.tight_layout()
        plt.savefig(fname+'pdf', transparent=True, bbox_inches='tight')

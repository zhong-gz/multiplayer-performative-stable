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
run_experiment = 0 # 1: run the experiment, 0: load the data
mu_A_list = [0.25,0.50,0.75,1.0] #25,0.50,0.75,1
gamma_list = [2.1,4,6,8,10]
BATCH=10
MAXITER=15
tot_rev=1

fs=44
lw=4
# 定义橙色色系的颜色映射
cmap = plt.get_cmap('winter')

# 定义不同的线形
linestyles = ['-', '--', '-.', ':', (0, (5, 5))]

# 事先定义字典来存储每个 gamma 值对应的颜色和线形
style_dict = {}
for i, gamma in enumerate(gamma_list):
    color = cmap((i + 1) / len(gamma_list))  # 从颜色映射中获取颜色
    linestyle = linestyles[i % len(linestyles)]  # 循环使用线形
    if gamma == 2.1:

        style_dict[gamma] = {'color': color, 'linestyle': linestyle, 'linewidth': lw}
    else:
        style_dict[gamma] = {'color': color, 'linestyle': linestyle, 'linewidth': lw}
para_sen_data = {}
total_revenue_stats = []
para_sen_data_path = 'ride_share/data/para_sen_data.npy'
if run_experiment:
    for mu_A in mu_A_list:
        print('Runing at mu_A',mu_A)
        mu_AC = mu_A/2
        total_avg_data = {}
        total_var_data = {}
        for p in [0,1,2,3,4]:  #,1,2,3,4
            print('  Runing at price_index',p)
            price_index = p
            # set up the game
            loc_lst_index=list(range(0,loc_cap))
            price_lst_index=list(range(0,5))
            price_start = price_index*5+10
            
            all_data={}
            for num_exper in range(num_experiments):
                seed = seed+1
                np.random.seed(seed)
                all_data[num_exper]={}
                params = {}
                params['A1']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
                params['A2']  = generate_negative_definite_matrix(loc_cap, diag_scale=mu_A)
                params['Ac1'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
                params['Ac2'] = generate_positive_definite_matrix(loc_cap, diag_scale=mu_AC)
                print('    Runing at number',num_exper+1,'trail')
                ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=False, params=params,maxx=10)
                ddgame.setup_distribution()

                # run all cases
                dic_data = {}
                for gamma in gamma_list:
                    print('      Runing at gamma:',gamma)
                    dic_data[gamma] = ddgame.runRR(gamma = gamma,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev)

                for gamma in gamma_list:
                    rev_key1 = f'revenue_total_p1'
                    rev_key2 = f'revenue_total_p2'
                    rev = np.asarray(dic_data[gamma][rev_key1]+dic_data[gamma][rev_key2])
                    all_data[num_exper][f'rev_{gamma}'] = rev

            avg_data = {}
            var_data = {} 
            for gamma in gamma_list:
                key = f'rev_{gamma}'
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

        para_sen_data[mu_A] = {'avg': total_avg_data, 'var': total_var_data}
    np.save(para_sen_data_path, para_sen_data)
else:
    para_sen_data = np.load(para_sen_data_path, allow_pickle=True).item()

fig, axes = plt.subplots(1, 4, figsize=figuresize)
# axes = axes.flatten()

# 找出所有子图数据在该大图对应的指标下的最小值和最大值
all_y_data = []
for i, mu_A in enumerate(mu_A_list):
    total_avg_data = para_sen_data[mu_A]['avg']
   
    for gamma in gamma_list:
        key = f'rev_{gamma}'
        all_y_data.extend(total_avg_data[key])

all_y_data = np.array(all_y_data)
if np.allclose(all_y_data, 0):
    power = 0
else:
    power = int(np.floor(np.log10(np.max(np.abs(all_y_data)))))
scale_factor = 10 ** power

y_min = min(all_y_data)/ scale_factor
y_max = max(all_y_data)/ scale_factor

for i, mu_A in enumerate(mu_A_list):
    total_avg_data = para_sen_data[mu_A]['avg']
    total_var_data = para_sen_data[mu_A]['var']
    ax = axes[i]

    for gamma in gamma_list:
        key = f'rev_{gamma}'
        total_rev = total_avg_data[key]/ scale_factor
        style = style_dict[gamma]
        ax.plot(total_rev, label=f'$\gamma$ = {gamma}', **style) #, color=color
        if i == 0:
            ax.set_ylabel(r'Total Revenue', fontsize=fs)
    axes[i].set_xlabel('Iterations', fontsize=fs)
    ax.set_title(f'$\mu_A = {mu_A}$', fontsize=fs) 
    ax.grid(True)
    ax.tick_params(labelsize=fs*0.7)
    # ax.set_ylim(y_min, y_max)
    tick_positions = np.arange(0, len(total_avg_data[f'rev_{gamma}'])+1, 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=fs*0.7)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.text(-0.1, 1.1, f'$\\times 10^{power}$', transform=ax.transAxes, fontsize=fs*0.7, verticalalignment='top')

handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=fs-2, ncol=len(gamma_list))
plt.tight_layout(rect=[0, 0.2, 1, 1])

plt.savefig(f'ride_share/figs/parameter_sensitivity.pdf', transparent=True, bbox_inches='tight')
plt.close()

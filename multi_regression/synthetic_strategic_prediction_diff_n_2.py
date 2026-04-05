import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
from scipy.linalg import eigh
import seaborn as sns
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
import random
import os
# import winsound

# Get number of CPU cores for parallel processing
N_JOBS = os.cpu_count() or 4  # Default to 4 if cpu_count() fails
print(f"Using {N_JOBS} CPU cores for parallel processing.")
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilssp_vector_map_diff import *
# %load_ext autoreload
# %autoreload 2

import time

def plot_time_comparison(sigma_A_values, figuresize, n, m, d, MAXITER, result_folder):
    """绘制运行时间对比图（以最快方法为横轴基准）"""
    fig, axes = plt.subplots(1, len(sigma_A_values), figsize=figuresize)
    if len(sigma_A_values) == 1:
        axes = [axes]
    
    handles = []
    labels = []
    
    for i, sigma_A in enumerate(sigma_A_values):
        sigma_AC = sum_A_AC - sigma_A
        sigma_C = sigma_A / n
        filepath = os.path.join(result_folder, f'figs_{MAXITER}/')
        file_name_npy = filepath + 'sig_A_' + str(sigma_A) + '_sigma_AC_' + str(sigma_AC) + '_m_' + str(m) + '_sigma_C_' + str(sigma_C) + '.npz'
        
        try:
            all_data = np.load(file_name_npy, allow_pickle=True)['all_data'].item()
        except:
            print(f"file {file_name_npy} not found, skipping...")
            continue
            
        # 收集所有种子的时间和误差数据
        time_data = {model: [] for model in ['agd', 'rgd', 'sfb', 'opgd', 'rr', 'sirr']}
        error_data = {model: [] for model in ['agd', 'rgd', 'sfb', 'opgd', 'rr', 'sirr']}
        
        for seed in seeds:
            for model in time_data.keys():
                if f'time_{model}' in all_data[seed] and f'error_{model}' in all_data[seed]:
                    # 获取累计时间序列（已经是累计时间）
                    cumulative_times = all_data[seed][f'time_{model}']
                    # 确保时间序列长度与误差序列一致
                    errors = all_data[seed][f'error_{model}']
                    
                    # 在时间序列前添加0（第0次迭代的时间为0）
                    # 确保所有方法的时间序列都从0开始
                    if len(cumulative_times) == len(errors) - 1:
                        # 时间序列比误差序列少1个点，需要添加0
                        cumulative_times = np.concatenate(([0], cumulative_times))
                    elif len(cumulative_times) == len(errors):
                        # 时间序列和误差序列长度相同，不需要添加
                        pass
                    else:
                        # 长度不匹配，使用最短长度
                        min_len = min(len(cumulative_times), len(errors))
                        cumulative_times = cumulative_times[:min_len]
                        errors = errors[:min_len]
                    
                    time_data[model].append(cumulative_times)
                    error_data[model].append(errors)
        
        # 计算均值和方差
        time_means = {}
        error_means = {}
        time_vars = {}
        error_vars = {}
        
        # 找到最快方法的最终时间
        min_final_time = float('inf')
        fastest_model = None
        # 在绘图之前添加：收集所有y轴数据
        all_y_data = []

        # 在计算每个模型的均值和方差后，收集y轴数据
        for model in time_means.keys():
            if model in error_means and len(error_means[model]) > 0:
                all_y_data.extend(error_means[model])

        # 计算y轴的最小值和最大值，并留出一些边距
        if all_y_data:
            y_min = np.min(all_y_data) * 0.8  # 留出20%的边距
            y_max = np.max(all_y_data) * 1.2  # 留出20%的边距
            
            # 确保y_min不为0（对数坐标需要正数）
            if y_min <= 0:
                y_min = np.min(all_y_data[all_y_data > 0]) * 0.5 if np.any(all_y_data > 0) else 0.1
            
            # 为每个子图设置统一的y轴范围
            for ax in axes:
                ax.set_ylim(y_min, y_max)

        
        for model in time_data.keys():
            if time_data[model] and len(time_data[model][0]) > 0:
                # 找到所有序列的最小长度
                min_length = min(len(t) for t in time_data[model])
                min_length = min(min_length, min(len(e) for e in error_data[model]))
                
                # 截取相同长度的序列
                time_truncated = [t[:min_length] for t in time_data[model]]
                error_truncated = [e[:min_length] for e in error_data[model]]
                
                # 计算均值和方差
                time_means[model] = np.mean(time_truncated, axis=0)
                error_means[model] = np.mean(error_truncated, axis=0)
                time_vars[model] = np.var(time_truncated, axis=0)
                error_vars[model] = np.var(error_truncated, axis=0)
                
                # 检查是否为最快方法
                final_time = time_means[model][-1]
                if final_time < min_final_time:
                    min_final_time = final_time
                    fastest_model = model
        
        # 对齐起点：将每个方法的时间序列减去其第一个时间点
        time_means_aligned = {}
        for model in time_means.keys():
            if len(time_means[model]) > 0:
                # 获取第一个时间点（即第一次迭代的运行时间）
                first_time = time_means[model][0]
                # 将整个时间序列减去第一个时间点，使起点对齐在0
                time_means_aligned[model] = time_means[model] - first_time
        
        # 绘制曲线
        models_plot = ['sirr', 'rr', 'rgd', 'sfb', 'agd', 'opgd']
        model_names = ['SIR$^2$', 'RR', 'RGD', 'SFB', 'AGM', 'OPGD']
        
        for j, model in enumerate(models_plot):
            if model in time_means_aligned and len(time_means_aligned[model]) > 0:
                # 获取样式
                style = style_dict[model_names[j]]
                
                # 绘制主曲线（使用对齐后的时间）
                line, = axes[i].plot(time_means_aligned[model], error_means[model], 
                                   label=model_names[j], **style)
            
                
                # 修正图例收集逻辑
                if i == 0:  # 只在第一个子图收集图例
                    if model_names[j] not in labels:  # 避免重复
                        handles.append(line)
                        labels.append(model_names[j])
        
        axes[i].set_title(f'$\sigma_{{a_i}}^2 = {sigma_A}$', fontsize=fs)
        axes[i].set_xlabel('Running Time', fontsize=fs)
        if i == 0:
            axes[i].set_ylabel('RMSE', fontsize=fs)
        axes[i].tick_params(labelsize=fs*0.7)
        axes[i].grid(True)
        axes[i].set_yscale('log')
        
        # 设置x轴上限为SIRR方法的7次迭代时长
        sirr_7_iter_time = time_means_aligned['sirr'][6] if 'sirr' in time_means_aligned and len(time_means_aligned['sirr']) > 6 else 0.003
        axes[i].set_xlim(0, sirr_7_iter_time)
    
    # 添加图例
    fig.legend(handles, labels, loc='lower center', ncol=len(models_plot), 
               fontsize=fs-2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.2, 1, 1])

    # fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=fs-2) #,bbox_to_anchor=(0.5, -0.02)
    # plt.tight_layout(rect=[0, 0.2, 1, 1])
    
    # 保存图片
    save_path = os.path.join(filepath, 'time_comparison_fair_multi_regression.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"time comparison figure saved to {save_path}")

    return fig, axes

start_time_total = time.time()

from itertools import product

seed = 42
np.random.seed(seed)
seeds= range(42,52)
run_experiment = 0 # 1: run the experiment, 0: load the data
sigma_theta= 0.1 ###
sigma_w=0.01

# Define parameter ranges for n, m, and d
n_values = [10,100]  # Modify to add more values: [2, 4, 6]
m_values = [10000]  # Modify to add more values: [50, 100, 200]
d_values = [10,100,1000]  # Modify to add more values: [5, 10, 20]

k = 10
sigma_A_values = [0.25, 0.5, 0.75, 1.0]
sigma_A_values = [x * k for x in sigma_A_values]
sum_A_AC = 1.25*k
eta=0.01
mu=1
nu0=1
alpha_rr = 0
models = ['SIR$^2$', 'RR','RGD','SFB','AGM','OPGD']
lw = 4
fs=40
figuresize = (25, 6)
style_dict = {
    'SIR$^2$': {'color': '#FF7F50', 'linestyle': '-', 'linewidth': lw+1},
    'RR': {'color': "#9b0000", 'linestyle': (0, (2, 2)), 'linewidth': lw},
    'AGM': {'color': '#9467bd', 'linestyle': '--', 'linewidth': lw},
    'RGD': {'color': '#444444', 'linestyle': ':', 'linewidth': lw},
    'SFB': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': lw},
    'OPGD': {'color': '#1f77b4', 'linestyle': (0, (5, 5)), 'linewidth': lw}
}

all_data_global={}
MAXITER= 100 #100 for RMSE comparison , 200 for time comparison

# Iterate over all combinations of n, m, and d
for n, m, d in product(n_values, m_values, d_values):
    print(f"\n{'='*60}")
    print(f"Running experiment for n={n}, m={m}, d={d}")
    print(f"{'='*60}")
    
    # Create result folder for current setting
    result_folder = f"multi_regression/results_n_{n}_m_{m}_d_{d}/"
    os.makedirs(result_folder, exist_ok=True)
    print(f"Result folder: {result_folder}")
    
    # Initialize B matrix for this setting
    B = np.random.normal(0,sigma_theta,size=(d,1))
    
    # Initialize lam for n players (vectorized)
    lam = [0.1] * n
    
    # Initialize all_data for this parameter combination
    all_data = {}
    
    for sigma_A in sigma_A_values:
        print('running sigma_A:',sigma_A)
        sigma_AC = sum_A_AC-sigma_A
        sigma_C = sigma_A/n
        filepath = os.path.join(result_folder, f'figs_{MAXITER}')
        os.makedirs(filepath, exist_ok=True)
        file_name_npy = os.path.join(filepath, f'sig_A_{sigma_A}_sigma_AC_{sigma_AC}_m_{m}_sigma_C_{sigma_C}.npz')
        
        # Create matrices for all n players (vectorized)
        A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
        Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
        C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
        
        # Create params dict with list format for compatibility
        params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}

        if run_experiment == 1:
            ddg=ddstrategic_prediction(MAXITER=MAXITER, sigma_theta=sigma_theta,sigma_w=sigma_w,
                                B=B,lam=lam,n=n, m=m, d=d, params=params,mu_w1=0, mu_w2=0, mu_theta=0)

            for seed in seeds:
                np.random.seed(seed)

                ## for AGM - initialize estimates for each player (vectorized)
                A_hats = [[np.zeros((1,d))] for _ in range(n)]
                Ac_hats = [[np.zeros((1,d))] for _ in range(n)]

                ## for OPGD - initialize expanded A matrices for each player (vectorized)
                A_opgd_list = [np.zeros((d+1,d)) for _ in range(n)]

                x0=np.random.uniform(size=(n,d))
                all_data[seed]={}
                all_data[seed]['x0']=x0

                x_sgd=[x0]
                x_agd=[x0]
                x_rgd=[x0]
                x_sfb=[x0]
                x_opgd=[x0]
                x_rr =[np.zeros((n,d))]
                x_sirr =[np.zeros((n,d))]
                rr_model = []
                sirr_model = []
                epsilon_i = 0
                gamma = 2.1
                alpha = 0
                count = 0

                # 在种子循环内，为每个算法添加时间记录
                all_data[seed]['time_agd'] = []
                all_data[seed]['time_rgd'] = []
                all_data[seed]['time_sfb'] = []
                all_data[seed]['time_opgd'] = []
                all_data[seed]['time_rr'] = []
                all_data[seed]['time_sirr'] = []

                # 初始化时间累计变量
                time_accum_agd = 0
                time_accum_rgd = 0
                time_accum_sfb = 0
                time_accum_opgd = 0
                time_accum_rr = 0
                time_accum_sirr = 0

                for i in range(MAXITER):
                    nu= 0.1 # nu is the learning rate for AGM
                    th=np.random.normal(0,sigma_theta,size=(d,m))
                    x_sgd.append(ddg.proj(x_sgd[-1]- eta*ddg.getgrad(x_sgd[-1],th)))
                    
                    ## for AGM (vectorized)
                    start_time = time.time()
                    Ahats_current = [A_hats[i_p][-1] for i_p in range(n)]
                    AChats_current = [Ac_hats[i_p][-1] for i_p in range(n)]
                    x_agd.append(ddg.proj(x_agd[-1]- 0.1*eta*ddg.getgrad_agd(x_agd[-1], th, Ahats=Ahats_current, AChats=AChats_current, passvals=True)))
                    
                    # Get z_list for update_estimate (vectorized)
                    z_list_agd, theta_agd = ddg.distribution_map(x_agd[-1], th)
                    Ahats_new, AChats_new = ddg.update_estimate(x_agd[-1], z_list_agd, theta_agd, nu=nu, mu=mu, Ahats=Ahats_current, AChats=AChats_current, passvals=True, UNCORR=False)
                    # Update all players at once (vectorized)
                    for i_p in range(n):
                        A_hats[i_p].append(Ahats_new[i_p])
                        Ac_hats[i_p].append(AChats_new[i_p])
                    
                    time_accum_agd += time.time() - start_time
                    all_data[seed]['time_agd'].append(time_accum_agd)

                    ## for rgd
                    start_time = time.time()
                    z_list,theta_rgd = ddg.distribution_map(x_rgd[-1],th)
                    x_rgd.append(ddg.proj(x_rgd[-1]-0.1*eta*ddg.getgrad_rgd(x_rgd[-1],z_list, theta_rgd)))
                    time_accum_rgd += time.time() - start_time
                    all_data[seed]['time_rgd'].append(time_accum_rgd)
                    ## for sfb
                    start_time = time.time()
                    z_list,theta_sfb = ddg.distribution_map(x_sfb[-1],th)
                    x_sfb.append(ddg.proj(x_sfb[-1]-(eta*(i+1)**(-3/4))*ddg.getgrad_rgd(x_sfb[-1],z_list, theta_sfb)))
                    time_accum_sfb += time.time() - start_time
                    all_data[seed]['time_sfb'].append(time_accum_sfb)
                    ## for OPGD
                    start_time = time.time()
                    x_opgd.append(ddg.proj(x_opgd[-1]-100*eta*(6/(10+i))*ddg.getgrad_opgd(x_opgd[-1],th,Ahats_opgd=A_opgd_list)))
                    z_list_opgd, theta_opgd = ddg.distribution_map(x_opgd[-1], th)
                    A_opgd_list = ddg.update_estimate_opgd(x_opgd[-1], z_list_opgd, theta_opgd, v_t=0.1*eta*7/((10+i)**(3/4)), Ahats_opgd=A_opgd_list)
                    time_accum_opgd += time.time() - start_time
                    all_data[seed]['time_opgd'].append(time_accum_opgd)

                    # for SIRR (vectorized + parallel)
                    start_time = time.time()
                    z_list_si,theta_t_1si = ddg.distribution_map(x_sirr[-1],th)
                    # 当维度>=数据量时加入正则项以避免矩阵病态
                    alpha_sirr = max(alpha, 1e-6) if d >= m else alpha
                    
                    # Ridge regression fitting for all players (no Parallel - too light for overhead)
                    sirr_models_t_1 = []
                    x_sirr_t = []
                    for i_player in range(n):
                        sirr_model_i = Ridge(alpha=alpha_sirr)
                        sirr_model_i.fit(theta_t_1si.T, z_list_si[i_player], sample_weight=1/m)
                        sirr_models_t_1.append(sirr_model_i)
                        x_sirr_t.append(sirr_model_i.coef_)
                    
                    x_sirr.append(np.vstack(x_sirr_t))
                    sirr_model.append(sirr_models_t_1)

                    z_list_tsi,theta_tsi = ddg.distribution_map(x_sirr[-1],th)
                    grad_diffs = []
                    
                    # Parallel gradient difference computation for all players
                    def compute_grad_diff_sirr(i_player):
                        g_t = -theta_tsi@(z_list_tsi[i_player]-theta_tsi.T@x_sirr[-1][i_player])/m
                        g_t_1 = -theta_t_1si@(z_list_si[i_player]-theta_t_1si.T@x_sirr[-1][i_player])/m
                        grad_diff = la.norm(g_t-g_t_1)
                        state_diff = la.norm(x_sirr[-1][i_player]-x_sirr[-2][i_player]+1e-3)
                        if la.norm(x_sirr[-1][i_player]-x_sirr[-2][i_player]) > 1e-3*d:
                            return grad_diff/state_diff
                        return None
                    
                    grad_diff_results = Parallel(n_jobs=N_JOBS, backend='loky')(
                        delayed(compute_grad_diff_sirr)(i_player) for i_player in range(n)
                    )
                    grad_diffs = [g for g in grad_diff_results if g is not None]
                    
                    if grad_diffs:
                        epsilon_i = max(epsilon_i, max(grad_diffs))
                        sqrt_epsilon = epsilon_i**2
                        alpha = gamma*sqrt_epsilon**0.5
                    time_accum_sirr += time.time() - start_time
                    all_data[seed]['time_sirr'].append(time_accum_sirr)

                    # for RR (vectorized, no parallel)
                    start_time = time.time()
                    z_list_rr,theta_t_1 = ddg.distribution_map(x_rr[-1],th)
                    # 当维度>=数据量时加入正则项以避免矩阵病态
                    alpha_rr_current = 1e-10 if (d >= m and alpha_rr == 0) else alpha_rr
                    
                    # Ridge regression fitting for all players (no Parallel - too light for overhead)
                    rr_models_t_1 = []
                    x_rr_t = []
                    for i_player in range(n):
                        rr_model_i = Ridge(alpha=alpha_rr_current)
                        rr_model_i.fit(theta_t_1.T, z_list_rr[i_player], sample_weight=1/m)
                        rr_models_t_1.append(rr_model_i)
                        x_rr_t.append(rr_model_i.coef_)
                    
                    x_rr.append(np.vstack(x_rr_t))
                    rr_model.append(rr_models_t_1)
                    time_accum_rr += time.time() - start_time
                    all_data[seed]['time_rr'].append(time_accum_rr)

                x_sgd=np.asarray(x_sgd)
                x_agd=np.asarray(x_agd)
                x_rgd=np.asarray(x_rgd)
                x_sfb=np.asarray(x_sfb)
                x_opgd=np.asarray(x_opgd)
                x_rr=np.asarray(x_rr)
                x_sirr=np.asarray(x_sirr)

                error_sgd=[]
                error_agd=[]
                error_rgd=[]
                error_sfb=[]
                error_opgd=[]
                error_rr=[]
                error_sirr=[]
                # estimate the loss (vectorized, no parallel - faster than threading overhead)
                th=1*np.random.normal(0,sigma_theta,size=(d,m))
                # th=1*np.random.uniform(size=(d,m))
                for x,y,z,sfb,rr_m,sirr_m,opgd in zip(x_sgd,x_agd,x_rgd,x_sfb,rr_model,sirr_model,x_opgd):
                    # Vectorized loss computation - no Parallel for speed
                    z_list_x, th_x = ddg.distribution_map(x,th)
                    loss_x = sum(la.norm(z_list_x[i]-th_x.T@x[i])**2 for i in range(n)) / (n*m)
                    error_sgd.append(loss_x)

                    z_list_y, th_y = ddg.distribution_map(y,th)
                    loss_y = sum(la.norm(z_list_y[i]-th_y.T@y[i])**2 for i in range(n)) / (n*m)
                    error_agd.append(loss_y)

                    z_list_z, th_z = ddg.distribution_map(z,th)
                    loss_z = sum(la.norm(z_list_z[i]-th_z.T@z[i])**2 for i in range(n)) / (n*m)
                    error_rgd.append(loss_z)

                    z_list_sfb, th_sfb = ddg.distribution_map(sfb,th)
                    loss_sfb = sum(la.norm(z_list_sfb[i]-th_sfb.T@sfb[i])**2 for i in range(n)) / (n*m)
                    error_sfb.append(loss_sfb)

                    z_list_opgd, th_opgd = ddg.distribution_map(opgd,th)
                    loss_opgd = sum(la.norm(z_list_opgd[i]-th_opgd.T@opgd[i])**2 for i in range(n)) / (n*m)
                    error_opgd.append(loss_opgd)

                    # RR loss computation - vectorized
                    rr = np.vstack((rr_m[0].coef_,) + tuple(rr_m[i].coef_ for i in range(1, n)))
                    z_list_rr, th_rr = ddg.distribution_map(rr,th)
                    loss_rr = sum(la.norm(z_list_rr[i]-rr_m[i].predict(th_rr.T))**2 for i in range(n)) / (n*m)
                    error_rr.append(loss_rr)

                    # SIRR loss computation - vectorized
                    sirr = np.vstack((sirr_m[0].coef_,) + tuple(sirr_m[i].coef_ for i in range(1, n)))
                    z_list_sirr, th_sirr = ddg.distribution_map(sirr,th)
                    loss_sirr = sum(la.norm(z_list_sirr[i]-sirr_m[i].predict(th_sirr.T))**2 for i in range(n)) / (n*m)
                    error_sirr.append(loss_sirr)

                err_agd=np.asarray(np.sqrt(error_agd))
                err_sgd=np.asarray(np.sqrt(error_sgd))
                err_rgd=np.asarray(np.sqrt(error_rgd))
                err_sfb=np.asarray(np.sqrt(error_sfb))
                err_opgd=np.asarray(np.sqrt(error_opgd))
                err_rr=np.asarray(np.sqrt(error_rr))
                err_sirr=np.asarray(np.sqrt(error_sirr))

                all_data[seed]['error_agd']=err_agd
                all_data[seed]['error_sgd']=err_sgd
                all_data[seed]['error_rgd']=err_rgd
                all_data[seed]['error_sfb']=err_sfb
                all_data[seed]['error_opgd']=err_opgd
                all_data[seed]['error_rr']=err_rr
                all_data[seed]['error_sirr']=err_sirr

            np.savez(file_name_npy, all_data=all_data)
            print(f"Data saved to {file_name_npy}")
        else:
            all_data = np.load(file_name_npy, allow_pickle=True)['all_data'].item()

    # Generate Plots for current (n, m, d) setting
    print(f"Generating plots for n={n}, m={m}, d={d}")
    for k in range(2):
        fig, axes = plt.subplots(1, 4, figsize=figuresize)
        axes = axes.flatten()
        handles = []
        labels = []
        all_y_data = []
        all_stats = []

        for i, sigma_A in enumerate(sigma_A_values):
            sigma_AC = sum_A_AC - sigma_A
            sigma_C = sigma_A / n
            fig_filepath = os.path.join(result_folder, f'figs_{MAXITER}')
            file_name_npy = os.path.join(fig_filepath, f'sig_A_{sigma_A}_sigma_AC_{sigma_AC}_m_{m}_sigma_C_{sigma_C}.npz')
            all_data = np.load(file_name_npy, allow_pickle=True)['all_data'].item()

            errs_agd = []
            errs_sgd = []
            errs_rgd = []
            errs_sfb = []
            errs_opgd = []
            errs_rr = []
            errs_sirr = []

            for seed in seeds:
                errs_agd.append(all_data[seed]['error_agd'])
                errs_sgd.append(all_data[seed]['error_sgd'])
                errs_rgd.append(all_data[seed]['error_rgd'])
                errs_sfb.append(all_data[seed]['error_sfb'])
                errs_opgd.append(all_data[seed]['error_opgd'])
                errs_rr.append(all_data[seed]['error_rr'])
                errs_sirr.append(all_data[seed]['error_sirr'])

            errs_agd = np.asarray(errs_agd)
            errs_sgd = np.asarray(errs_sgd)
            errs_rgd = np.asarray(errs_rgd)
            errs_sfb = np.asarray(errs_sfb)
            errs_opgd = np.asarray(errs_opgd)
            errs_rr = np.asarray(errs_rr)
            errs_sirr = np.asarray(errs_sirr)

            errs_agd_mean = np.mean(errs_agd, axis=0)
            errs_sgd_mean = np.mean(errs_sgd, axis=0)
            errs_rgd_mean = np.mean(errs_rgd, axis=0)
            errs_sfb_mean = np.mean(errs_sfb, axis=0)
            errs_opgd_mean = np.mean(errs_opgd, axis=0)
            errs_rr_mean = np.mean(errs_rr, axis=0)
            errs_sirr_mean = np.mean(errs_sirr, axis=0)

            errs_agd_var = np.var(errs_agd, axis=0)
            errs_sgd_var = np.var(errs_sgd, axis=0)
            errs_rgd_var = np.var(errs_rgd, axis=0)
            errs_sfb_var = np.var(errs_sfb, axis=0)
            errs_opgd_var = np.var(errs_opgd, axis=0)
            errs_rr_var = np.var(errs_rr, axis=0)
            errs_sirr_var = np.var(errs_sirr, axis=0)

            stat_str = f'{errs_sirr_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_sirr_var[-1]):0.4f}'
            all_stats.append({'model': 'SIR$^2$','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})
            stat_str = f'{errs_rr_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_rr_var[-1]):0.4f}'
            all_stats.append({'model': 'RR','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})
            stat_str = f'{errs_rgd_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_rgd_var[-1]):0.4f}'
            all_stats.append({'model': 'RGD','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})
            stat_str = f'{errs_sfb_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_sfb_var[-1]):0.4f}'
            all_stats.append({'model': 'SFB','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})
            stat_str = f'{errs_agd_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_agd_var[-1]):0.4f}'
            all_stats.append({'model': 'AGM','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})
            stat_str = f'{errs_opgd_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_opgd_var[-1]):0.4f}'
            all_stats.append({'model': 'OPGD','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})

            iterations = np.arange(0, MAXITER)

            if k == 0:
                axes[i].fill_between(iterations, errs_rgd_mean - np.sqrt(errs_rgd_var), errs_rgd_mean + np.sqrt(errs_rgd_var), alpha=0.2, color=style_dict['RGD']['color'],edgecolor='none')
                axes[i].fill_between(iterations, errs_sfb_mean - np.sqrt(errs_sfb_var), errs_sfb_mean + np.sqrt(errs_sfb_var), alpha=0.2, color=style_dict['SFB']['color'],edgecolor='none')
                axes[i].fill_between(iterations, errs_agd_mean - np.sqrt(errs_agd_var), errs_agd_mean + np.sqrt(errs_agd_var), alpha=0.2, color=style_dict['AGM']['color'],edgecolor='none')
                axes[i].fill_between(iterations, errs_opgd_mean - np.sqrt(errs_opgd_var), errs_opgd_mean + np.sqrt(errs_opgd_var), alpha=0.2, color=style_dict['OPGD']['color'],edgecolor='none')
                axes[i].fill_between(iterations, errs_rr_mean - np.sqrt(errs_rr_var), errs_rr_mean + np.sqrt(errs_rr_var), alpha=0.2, color=style_dict['RR']['color'],edgecolor='none')
                axes[i].fill_between(iterations, errs_sirr_mean - np.sqrt(errs_sirr_var), errs_sirr_mean + np.sqrt(errs_sirr_var), alpha=0.2, color=style_dict['SIR$^2$']['color'],edgecolor='none')
            
            l1, = axes[i].plot(errs_rgd_mean, label='RGD',**style_dict['RGD'])
            l3, = axes[i].plot(errs_sfb_mean, label='SFB',**style_dict['SFB'])
            l2, = axes[i].plot(errs_agd_mean, label='AGM',**style_dict['AGM'])
            l4, = axes[i].plot(errs_opgd_mean, label='OPGD',**style_dict['OPGD'])
            l5, = axes[i].plot(errs_rr_mean, label='RR',**style_dict['RR'])
            l6, = axes[i].plot(errs_sirr_mean, label='SIR$^2$',**style_dict['SIR$^2$'])
            if i == 0:
                axes[i].set_ylabel('RMSE', fontsize=fs)
                handles.extend([l6,l5, l1, l3, l2,l4])
                labels.extend(['SIR$^2$', 'RR','RGD', 'SFB', 'AGM', 'OPGD'])

            axes[i].set_title(f'$\sigma_{{a_i}}^2 = {sigma_A}$', fontsize=fs)
            axes[i].set_xlabel('Iterations', fontsize=fs)
            axes[i].tick_params(labelsize=fs*0.7)
            axes[i].grid(True)
            axes[i].set_yscale('log')
            all_y_data.extend([errs_sirr_mean,errs_rr_mean, errs_rgd_mean, errs_agd_mean, errs_sfb_mean, errs_opgd_mean])
        
        # 找出所有 y 数据的最小值和最大值
        all_y_data = np.concatenate(all_y_data)
        y_min = np.min(all_y_data)- 0.2*np.abs(np.min(all_y_data))
        y_max = np.max(all_y_data)

        # 统一所有子图的 y 轴范围
        for ax in axes:
            ax.set_ylim(y_min, y_max)

        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=fs-2) #,bbox_to_anchor=(0.5, -0.02)
        plt.tight_layout(rect=[0, 0.2, 1, 1])  # 底部留出 10% 的空间

        # 保存图片
        if not os.path.exists(fig_filepath):
            os.makedirs(fig_filepath)
        if k == 0:
            save_path = os.path.join(fig_filepath, 'combined_plot_multi_regression_var.pdf')
        else:
            save_path = os.path.join(fig_filepath, 'combined_plot_multi_regression.pdf')
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")

        # 创建 DataFrame 并保存为 Excel
        df = pd.DataFrame(all_stats)
        pivot_df = df.pivot(index='model', columns='sigma_A', values='result')
        # 确保 model 顺序和列表一致
        pivot_df = pivot_df.reindex(models)
        excel_path = os.path.join(result_folder, f'figs_{MAXITER}', 'regression_result.xlsx')
        pivot_df.to_excel(excel_path)
        print(f"Excel result saved to {excel_path}")

    # 生成时间对比图
    print(f"Generating time comparison plots for n={n}, m={m}, d={d}")
    plot_time_comparison(sigma_A_values, figuresize=(25, 5), n=n, m=m, d=d, MAXITER=MAXITER, result_folder=result_folder)

end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time_total  # 计算耗时（秒）

print(f"The time of this code need to run: {execution_time:.4f} 秒")


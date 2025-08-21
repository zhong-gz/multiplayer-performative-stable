import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
import seaborn as sns
from sklearn.linear_model import Ridge
import random
import winsound
import sys, os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utils.utilssp_vector_map import *
# %load_ext autoreload
# %autoreload 2

import time

start_time = time.time()

seed = 42
np.random.seed(seed)
seeds= range(42,52)
run_experiment = 1 # 1: run the experiment, 0: load the data
sigma_theta= 0.1 ###
sigma_w=0.0001
nu=1e-3
n=2
m= 100 # both players dimension of z_i
d=2 # size of each players action
B = np.random.normal(0,sigma_theta,size=(d,1))
sigma_A_values = [0.25, 0.5, 0.75, 1.0]
eta=0.1
mu=2
nu0=1
models = ['SIR$^2$', 'RGD','SFB','AGM','OPGD']
lw = 4
fs=40
figuresize = (25, 5)
style_dict = {
    'SIR$^2$': {'color': '#FF7F50', 'linestyle': '-', 'linewidth': lw+1},
    'AGM': {'color': '#9467bd', 'linestyle': '--', 'linewidth': lw},
    'RGD': {'color': '#444444', 'linestyle': ':', 'linewidth': lw},
    'SFB': {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': lw},
    'OPGD': {'color': '#1f77b4', 'linestyle': (0, (5, 5)), 'linewidth': lw}
}

all_data={}
# lam=[1.0,1.0]
lam=[0.5,0.5]
MAXITER=100

for sigma_A in sigma_A_values:
    print('running sigma_A:',sigma_A)
    sigma_AC = 1.25-sigma_A
    sigma_C = sigma_A/n
    filepath = 'multi_regression/figs_'+str(MAXITER)+'/'
    file_name_npy = filepath+'sig_A_'+str(sigma_A)+'_sigma_AC_'+str(sigma_AC)+'_m_'+str(m)+'_sigma_C_'+str(sigma_C)+'.npz'
    filename=filepath+'sig_A_'+str(sigma_A)+'_sigma_AC_'+str(sigma_AC)+'_m_'+str(m)+'_sigma_C_'+str(sigma_C)+'.'
    A1= np.random.normal(0,np.sqrt(sigma_A),size=(1,d))
    Ac1= np.random.normal(0,np.sqrt(sigma_AC),size=(1,d))
    A2= np.random.normal(0,np.sqrt(sigma_A),size=(1,d))
    Ac2= np.random.normal(0,np.sqrt(sigma_AC),size=(1,d))
    C1= np.random.normal(0,np.sqrt(sigma_C),size=(d,d))
    C2= np.random.normal(0,np.sqrt(sigma_C),size=(d,d))
    params={'A1':A1,'A2':A2,'Ac1':Ac1,'Ac2':Ac2,'C1':C1,'C2':C2}

    if run_experiment == 1:
        ddg=ddstrategic_prediction(MAXITER=MAXITER, sigma_theta=sigma_theta,sigma_w=sigma_w,
                            B=B,nu=nu, lam=lam,n=n, m=m, d=d, params=params,
                                mu_w1=0, mu_w2=0, mu_theta=0)

        for seed in seeds:
            np.random.seed(seed)

            ## for AGM
            A1_hat = np.zeros((1,d)) 
            Ac1_hat =np.zeros((1,d))
            A2_hat = np.zeros((1,d)) 
            Ac2_hat = np.zeros((1,d)) 
            A_dic={'A1_hats':[A1_hat], 'Ac1_hats': [Ac1_hat],'A2_hats': [A2_hat], 'Ac2_hats': [Ac2_hat] }

            ## for OPGD
            A1_opgd = np.zeros((d+1,d)) 
            A2_opgd = np.zeros((d+1,d))
            A_opgd_dic={'A1_opgds':[A1_opgd], 'A2_opgds': [A2_opgd]}

            x0=np.random.uniform(size=(2,d))
            all_data[seed]={}
            all_data[seed]['x0']=x0

            x_sgd=[x0]
            x_agd=[x0]
            x_rgd=[x0]
            x_sfb=[x0]
            x_opgd=[x0]
            x_rr =[np.zeros((2,d))]
            rr_model = []
            epsilon_1 = 0
            epsilon_2 = 0
            gamma = 2.1
            alpha = 0
            count = 0

            for i in range(MAXITER):
                nu=2*nu0/(len(x_agd)+2*3*d)
                # th=1*np.random.uniform(size=(d,m))
                th=np.random.normal(0,sigma_theta,size=(d,m))
                z1=ddg.D_w(0)
                z2=ddg.D_w(1)
                x_sgd.append(ddg.proj(x_sgd[-1]-eta*ddg.getgrad(x_sgd[-1],th)))
                ## for AGM
                x_agd.append(ddg.proj(x_agd[-1]-0.1*eta*ddg.getgrad_agd(x_agd[-1],th,A1hat=A_dic['A1_hats'][-1],Ac1hat=A_dic['Ac1_hats'][-1],
                                                                    A2hat=A_dic['A2_hats'][-1], Ac2hat=A_dic['Ac2_hats'][-1], passvals=True)))
                A1_hat,Ac1_hat,A2_hat,Ac2_hat = ddg.update_estimate(x_agd[-1], z1, z2,th,nu=nu,mu=mu, A1hat=A_dic['A1_hats'][-1],Ac1hat=A_dic['Ac1_hats'][-1],
                                                                    A2hat=A_dic['A2_hats'][-1], Ac2hat=A_dic['Ac2_hats'][-1], passvals=True,UNCORR=False)
                A_dic['A1_hats'].append(A1_hat)
                A_dic['Ac1_hats'].append(Ac1_hat)
                A_dic['A2_hats'].append(A2_hat)
                A_dic['Ac2_hats'].append(Ac2_hat)
                ## for rgd
                z1,z2,theta_rgd = ddg.distribution_map(x_rgd[-1],th)
                x_rgd.append(ddg.proj(x_rgd[-1]-eta*ddg.getgrad_rgd(x_rgd[-1],z1,z2, theta_rgd)))
                ## for sfb
                z1,z2,theta_sfb = ddg.distribution_map(x_sfb[-1],th)
                x_sfb.append(ddg.proj(x_sfb[-1]-(eta*(i+1)**(-3/4))*ddg.getgrad_rgd(x_sfb[-1],z1,z2, theta_sfb)))
                ## for OPGD
                x_opgd.append(ddg.proj(x_opgd[-1]-eta*(6/(10+i))*ddg.getgrad_opgd(x_opgd[-1],th,A1hat=A1_opgd, A2hat=A2_opgd)))
                A1_opgd, A2_opgd = ddg.update_estimate_opgd(x_opgd[-1], z1, z2,th,v_t = 0.1*eta*7/((10+i)**(3/4)), A1hat=A1_opgd, A2hat=A2_opgd)

                # for SIRR
                z1_t_1,z2_t_1,theta_t_1 = ddg.distribution_map(x_rr[-1],th)
                rr_model_1 = Ridge(alpha = alpha)
                rr_model_1.fit(theta_t_1.T,z1_t_1,sample_weight=1/m)
                rr_model_2 = Ridge(alpha = alpha)
                rr_model_2.fit(theta_t_1.T,z2_t_1,sample_weight=1/m)
                x_rr.append(np.vstack((rr_model_1.coef_,rr_model_2.coef_)))
                rr_model.append([rr_model_1,rr_model_2])

                z1_t,z2_t,theta_t = ddg.distribution_map(x_rr[-1],th)
                g1_t=-theta_t@(z1_t-theta_t.T@x_rr[-1][0])/m
                g2_t=-theta_t@(z2_t-theta_t.T@x_rr[-1][1])/m
                g1_t_1=-theta_t_1@(z1_t_1-theta_t_1.T@x_rr[-1][0])/m
                g2_t_1=-theta_t_1@(z2_t_1-theta_t_1.T@x_rr[-1][1])/m
                if (la.norm(x_rr[-1][0]-x_rr[-2][0]) > 1e-3 or la.norm(x_rr[-1][1]-x_rr[-2][1]) > 1e-3):
                    epsilon_1 = max(epsilon_1,la.norm(g1_t-g1_t_1)/(la.norm(x_rr[-1][0]-x_rr[-2][0])+1e-3))
                    epsilon_2 = max(epsilon_2,la.norm(g2_t-g2_t_1)/(la.norm(x_rr[-1][1]-x_rr[-2][1])+1e-3))
                    alpha = gamma*((epsilon_1**2+epsilon_2**2)**0.5)

            x_sgd=np.asarray(x_sgd)
            x_agd=np.asarray(x_agd)
            x_rgd=np.asarray(x_rgd)
            x_sfb=np.asarray(x_sfb)
            x_opgd=np.asarray(x_opgd)
            x_rr=np.asarray(x_rr)

            error_sgd=[]
            error_agd=[]
            error_rgd=[]
            error_sfb=[]
            error_opgd=[]
            error_rr=[]

            # estimate the loss
            th=1*np.random.normal(0,sigma_theta,size=(d,m))
            # th=1*np.random.uniform(size=(d,m))
            for x,y,z,sfb,rr_m,opgd in zip(x_sgd,x_agd,x_rgd,x_sfb,rr_model,x_opgd):
                z1,z2,th_x = ddg.distribution_map(x,th)
                error_sgd.append((la.norm(z1-th_x.T@x[0])**2 + la.norm(z2-th_x.T@x[1])**2)/(2*m))

                z1,z2,th_y = ddg.distribution_map(y,th)
                error_agd.append((la.norm(z1-th_y.T@y[0])**2 + la.norm(z2-th_y.T@y[1])**2)/(2*m))

                z1,z2,th_z = ddg.distribution_map(z,th)
                error_rgd.append((la.norm(z1-th_z.T@z[0])**2 + la.norm(z2-th_z.T@z[1])**2)/(2*m))

                z1,z2,th_sfb = ddg.distribution_map(sfb,th)
                error_sfb.append((la.norm(z1-th_sfb.T@sfb[0])**2 + la.norm(z2-th_sfb.T@sfb[1])**2)/(2*m))

                z1,z2,th_opgd = ddg.distribution_map(opgd,th)
                error_opgd.append((la.norm(z1-th_opgd.T@opgd[0])**2 + la.norm(z2-th_opgd.T@opgd[1])**2)/(2*m))

                rr = np.vstack((rr_m[0].coef_,rr_m[1].coef_))
                z1,z2,th_rr = ddg.distribution_map(rr,th)
                error_rr.append((la.norm(z1-rr_m[0].predict(th_rr.T))**2 + la.norm(z2-rr_m[1].predict(th_rr.T))**2)/(2*m))

            err_agd=np.asarray(np.sqrt(error_agd))
            err_sgd=np.asarray(np.sqrt(error_sgd))
            err_rgd=np.asarray(np.sqrt(error_rgd))
            err_sfb=np.asarray(np.sqrt(error_sfb))
            err_opgd=np.asarray(np.sqrt(error_opgd))
            err_rr=np.asarray(np.sqrt(error_rr))

            all_data[seed]['error_agd']=err_agd
            all_data[seed]['error_sgd']=err_sgd
            all_data[seed]['error_rgd']=err_rgd
            all_data[seed]['error_sfb']=err_sfb
            all_data[seed]['error_opgd']=err_opgd
            all_data[seed]['error_rr']=err_rr

        np.savez(file_name_npy, all_data=all_data)
        print(f"Data saved to {file_name_npy}")
    else:
        all_data = np.load(file_name_npy, allow_pickle=True)['all_data'].item()

# Generate Plots
print("Generating combined figure")
for k in range(2):
    fig, axes = plt.subplots(1, 4, figsize=figuresize)
    axes = axes.flatten()
    handles = []
    labels = []
    all_y_data = []
    all_stats = []


    for i, sigma_A in enumerate(sigma_A_values):
        sigma_AC = 1.25 - sigma_A
        sigma_C = sigma_A / n
        filepath = 'multi_regression/figs_' + str(MAXITER) + '/'
        file_name_npy = filepath + 'sig_A_' + str(sigma_A) + '_sigma_AC_' + str(sigma_AC) + '_m_' + str(m) + '_sigma_C_' + str(sigma_C) + '.npz'
        all_data = np.load(file_name_npy, allow_pickle=True)['all_data'].item()

        errs_agd = []
        errs_sgd = []
        errs_rgd = []
        errs_sfb = []
        errs_opgd = []
        errs_rr = []

        for seed in seeds:
            errs_agd.append(all_data[seed]['error_agd'])
            errs_sgd.append(all_data[seed]['error_sgd'])
            errs_rgd.append(all_data[seed]['error_rgd'])
            errs_sfb.append(all_data[seed]['error_sfb'])
            errs_opgd.append(all_data[seed]['error_opgd'])
            errs_rr.append(all_data[seed]['error_rr'])

        errs_agd = np.asarray(errs_agd)
        errs_sgd = np.asarray(errs_sgd)
        errs_rgd = np.asarray(errs_rgd)
        errs_sfb = np.asarray(errs_sfb)
        errs_opgd = np.asarray(errs_opgd)
        errs_rr = np.asarray(errs_rr)

        errs_agd_mean = np.mean(errs_agd, axis=0)
        errs_sgd_mean = np.mean(errs_sgd, axis=0)
        errs_rgd_mean = np.mean(errs_rgd, axis=0)
        errs_sfb_mean = np.mean(errs_sfb, axis=0)
        errs_opgd_mean = np.mean(errs_opgd, axis=0)
        errs_rr_mean = np.mean(errs_rr, axis=0)

        errs_agd_var = np.var(errs_agd, axis=0)
        errs_sgd_var = np.var(errs_sgd, axis=0)
        errs_rgd_var = np.var(errs_rgd, axis=0)
        errs_sfb_var = np.var(errs_sfb, axis=0)
        errs_opgd_var = np.var(errs_opgd, axis=0)
        errs_rr_var = np.var(errs_rr, axis=0)

        stat_str = f'{errs_rr_mean[-1]:0.4f} $\pm$ {np.sqrt(errs_rr_var[-1]):0.4f}'
        all_stats.append({'model': 'SIR$^2$','sigma_A': f'$\sigma_A$ = {sigma_A}','result': stat_str})
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
            axes[i].fill_between(iterations, errs_rr_mean - np.sqrt(errs_rr_var), errs_rr_mean + np.sqrt(errs_rr_var), alpha=0.2, color=style_dict['SIR$^2$']['color'],edgecolor='none')
        l1, = axes[i].plot(errs_rgd_mean, label='RGD',**style_dict['RGD'])
        l3, = axes[i].plot(errs_sfb_mean, label='SFB',**style_dict['SFB'])
        l2, = axes[i].plot(errs_agd_mean, label='AGM',**style_dict['AGM'])
        l4, = axes[i].plot(errs_opgd_mean, label='OPGD',**style_dict['OPGD'])
        l5, = axes[i].plot(errs_rr_mean, label='SIR$^2$',**style_dict['SIR$^2$'])
        if i == 0:
            axes[i].set_ylabel('RMSE', fontsize=fs)
            handles.extend([l5, l1, l3, l2,l4])
            labels.extend(['SIR$^2$', 'RGD', 'SFB', 'AGM', 'OPGD'])

        axes[i].set_title(f'$\sigma_{{a_i}}^2 = {sigma_A}$', fontsize=fs)
        axes[i].set_xlabel('Iterations', fontsize=fs)
        axes[i].tick_params(labelsize=fs*0.7)
        axes[i].grid(True)
        axes[i].set_yscale('log')
        all_y_data.extend([errs_rr_mean, errs_rgd_mean, errs_agd_mean, errs_sfb_mean, errs_opgd_mean])
    # 找出所有 y 数据的最小值和最大值
    all_y_data = np.concatenate(all_y_data)
    y_min = np.min(all_y_data)
    y_max = np.max(all_y_data)

    # 统一所有子图的 y 轴范围
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=fs-2) #,bbox_to_anchor=(0.5, -0.02)
    plt.tight_layout(rect=[0, 0.2, 1, 1])  # 底部留出 10% 的空间

    # 保存图片
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if k == 0:
        save_path = os.path.join(filepath, 'combined_plot_multi_regression_var.pdf')
    else:
        save_path = os.path.join(filepath, 'combined_plot_multi_regression.pdf')
    plt.savefig(save_path)
    print(f"Combined plot saved to {save_path}")


# 创建 DataFrame 并保存为 Excel
df = pd.DataFrame(all_stats)
pivot_df = df.pivot(index='model', columns='sigma_A', values='result')
# 确保 model 顺序和列表一致
pivot_df = pivot_df.reindex(models)
pivot_df.to_excel('multi_regression/figs_100/regression_result.xlsx')

end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算耗时（秒）

print(f"The time of this code need to run: {execution_time:.4f} 秒")
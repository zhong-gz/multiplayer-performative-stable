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
from utilssp_vector_map import *
# %load_ext autoreload
# %autoreload 2

seed = 42
np.random.seed(seed)
seeds= range(42,52)
sigma_theta= 0.1 ###
sigma_w=0.0001
nu=1e-3
n=2
m= 100 # both players dimension of z_i
d=2 # size of each players action
B = np.random.normal(0,sigma_theta,size=(d,1))

# lam=[1.0,1.0]
lam=[0,0]

sigma_A = 1.0
sigma_AC = 1.25-sigma_A
sigma_C = sigma_A/n
A1= np.random.normal(0,np.sqrt(sigma_A),size=(1,d))
Ac1= np.random.normal(0,np.sqrt(sigma_AC),size=(1,d))
A2= np.random.normal(0,np.sqrt(sigma_A),size=(1,d))
Ac2= np.random.normal(0,np.sqrt(sigma_AC),size=(1,d))
C1= np.random.normal(0,np.sqrt(sigma_C),size=(d,d))
C2= np.random.normal(0,np.sqrt(sigma_C),size=(d,d))
params={'A1':A1,'A2':A2,'Ac1':Ac1,'Ac2':Ac2,'C1':C1,'C2':C2}

MAXITER=1000
ddg=ddstrategic_prediction(MAXITER=MAXITER, sigma_theta=sigma_theta,sigma_w=sigma_w,
                       B=B,nu=nu, lam=lam,n=n, m=m, d=d, params=params,
                           mu_w1=0, mu_w2=0, mu_theta=0)

eta=0.1

mu=2
nu0=1
all_data={}

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
        x_opgd.append(ddg.proj(x_opgd[-1]-1*eta*(6/(10+i))*ddg.getgrad_opgd(x_opgd[-1],th,A1hat=A1_opgd, A2hat=A2_opgd)))
        A1_opgd, A2_opgd = ddg.update_estimate_opgd(x_opgd[-1], z1, z2,th,v_t = 0.1*eta*7/((10+i)**(3/4)), A1hat=A1_opgd, A2hat=A2_opgd)

        # for repeated retraining
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
        epsilon_1 = max(epsilon_1,la.norm(g1_t-g1_t_1)/la.norm(x_rr[-1][0]-x_rr[-2][0]))
        epsilon_2 = max(epsilon_2,la.norm(g2_t-g2_t_1)/la.norm(x_rr[-1][0]-x_rr[-2][0]))
        if (la.norm(x_rr[-1][0]-x_rr[-2][0]) > 1e-3 or la.norm(x_rr[-1][1]-x_rr[-2][1]) > 1e-3):
            count += 1
            if count < 10:
                alpha = gamma*((epsilon_1**2+epsilon_2**2)**0.5) #*0.01 + 0.99*alpha


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

filepath = './ppw_figs_log/'
file_name_npy = filepath+'ppw_sig_A_'+str(sigma_A)+'_sigma_AC_'+str(sigma_AC)+'_m_'+str(m)+'_sigma_C_'+str(sigma_C)+'.npz'
np.savez(file_name_npy, all_data=all_data)
print(f"Data saved to {file_name_npy}")

## Generate Plots
filename=filepath+'ppw_sig_A_'+str(sigma_A)+'_sigma_AC_'+str(sigma_AC)+'_m_'+str(m)+'_sigma_C_'+str(sigma_C)+'.'
print(f"Figure plot to {filename}")
SAVE=1

errs_agd=[]
errs_sgd=[]
errs_dfo=[]
errs_rgd=[]
errs_sfb=[]
errs_opgd=[]
errs_rr = []
fs=22
for seed in seeds:
    errs_agd.append(all_data[seed]['error_agd'])
    errs_sgd.append(all_data[seed]['error_sgd'])
    errs_rgd.append(all_data[seed]['error_rgd'])
    errs_sfb.append(all_data[seed]['error_sfb'])
    errs_opgd.append(all_data[seed]['error_opgd'])
    errs_rr.append(all_data[seed]['error_rr'])

errs_agd=np.asarray(errs_agd)
errs_sgd=np.asarray(errs_sgd)
errs_rgd=np.asarray(errs_rgd)
errs_sfb=np.asarray(errs_sfb)
errs_opgd=np.asarray(errs_opgd)
errs_rr=np.asarray(errs_rr)


errs_agd_mean=np.mean(errs_agd,axis=0)
errs_sgd_mean=np.mean(errs_sgd,axis=0)
errs_rgd_mean=np.mean(errs_rgd,axis=0)
errs_sfb_mean=np.mean(errs_sfb,axis=0)
errs_opgd_mean=np.mean(errs_opgd,axis=0)
errs_rr_mean=np.mean(errs_rr,axis=0)

errs_agd_var=np.std(errs_agd,axis=0)
errs_sgd_var=np.std(errs_sgd,axis=0)
errs_rgd_var=np.std(errs_rgd,axis=0)
errs_sfb_var=np.std(errs_sfb,axis=0)
errs_opgd_var=np.std(errs_opgd,axis=0)
errs_rr_var=np.std(errs_rr,axis=0)
# print(np.shape(errs_agd_var))

iterations=np.arange(0,MAXITER+1)
fig=plt.figure(figsize=(10,7))
# plt.title(f'mu_A:{mu_A:.1f},mu_AC:{mu_AC:.1f},m:{m:.0f}') 
# for i in range(len(errs_agd)):
#     # plt.plot(errs_sgd[i,:], linewidth=3,color='xkcd:cerulean', alpha=0.1)
#     plt.plot(errs_agd[i,:], linewidth=3, alpha=0.1, color='xkcd:light orange')
#     plt.plot(errs_rgd[i,:], linewidth=3, alpha=0.1, color='xkcd:light green')
    

# plt.plot(errs_sgd_mean, linewidth=3,color='xkcd:cerulean', label='SGM')
plt.plot(errs_rgd_mean, linewidth=3, color='xkcd:light green', label='RGD')
plt.plot(errs_agd_mean, linewidth=3, color='xkcd:light orange', label='AGM')
plt.plot(errs_sfb_mean, linewidth=3, color='xkcd:light red', label='SFB')
plt.plot(errs_opgd_mean, linewidth=3, color='xkcd:light purple', label='OPGD')
plt.plot(errs_rr_mean, linewidth=3, color='xkcd:light blue', label='Ours_RR')
# plt.fill_between(iterations,errs_sgd_mean+errs_sgd_var,errs_sgd_mean-errs_sgd_var, alpha=0.5, linewidth=0,color='xkcd:cerulean')
# plt.fill_between(iterations,errs_agd_mean+errs_agd_var,errs_agd_mean-errs_agd_var, alpha=0.4, linewidth=0, color='xkcd:light orange')
# plt.fill_between(iterations,errs_rgd_mean+errs_rgd_var,errs_rgd_mean-errs_rgd_var, alpha=0.4, linewidth=0, color='xkcd:light green')
# plt.fill_between(iterations,errs_rr_mean+errs_rr_var,errs_rr_mean-errs_rr_var, alpha=0.4, linewidth=0, color='xkcd:light blue')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)

plt.tick_params(labelsize=fs-2)
# plt.ylabel(r'$\mathbb{E}\sum_{i=1}^n \Vert z_i^t- \theta^\top x_i^t\Vert^2$', fontsize=fs)
plt.ylabel(r'RMSE', fontsize=fs)
plt.ylim([2*1e-3,3])
plt.xlabel(r'iterations', fontsize=fs)
plt.legend(fontsize=fs-2, loc='upper right',ncol=1)
if SAVE:
    for tag in ['pdf']:
        plt.savefig(filename+tag,  bbox_inches='tight', dpi=300)

# winsound.Beep(1500, 500)
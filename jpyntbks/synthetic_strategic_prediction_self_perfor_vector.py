import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
import seaborn as sns
from sklearn.linear_model import Ridge
import random
import sys, os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilssp_vector_map import *
# %load_ext autoreload
# %autoreload 2

seed = 42
np.random.seed(seed)
seeds= range(42,92)
sigma_theta= 0.1 ###
sigma_w=0.01
density=1 ###
nu=1e-3
m=100 # both players dimension of z_i
d=2 # size of each players action
# B=np.ones((d,1)) #np.array([[1],[1]]) #np.random.rand(d,1)
B=np.random.rand(d,1) ###
lam=[1.0,1.0]
# lam=[0.1,0.1]
# lam=[0,0]
mu_A = 1
mu_AC = 2

A1=mu_A*scirand(1,d,density=density).A ###
Ac1=mu_AC*scirand(1,d,density=density).A ###
A2=mu_A*scirand(1,d,density=density).A ###
Ac2=mu_AC*scirand(1,d,density=density).A  ###
params={'A1':A1,'A2':A2,'Ac1':Ac1,'Ac2':Ac2} 

MAXITER=1000
n=2
ddg=ddstrategic_prediction(MAXITER=MAXITER, sigma_theta=sigma_theta,sigma_w=sigma_w,density=density,
                       B=B,nu=nu, lam=lam,n=n, m=m, d=d, params=params,
                           mu_w1=0, mu_w2=0, mu_theta=0)

_,S1,_=la.svd(A1)
_,S2,_=la.svd(A2)
S1=np.sort(S1)[-1]
S2=np.sort(S2)[-1]
eta=0.01 #1/np.max([S1,S2])

A1_hat = np.zeros((1,d)) #np.random.rand(m,d)
Ac1_hat =np.zeros((1,d))# np.random.rand(m,d)
A2_hat = np.zeros((1,d)) #np.random.rand(m,d)
Ac2_hat = np.zeros((1,d)) #np.random.rand(m,d)

A_dic={'A1_hats':[A1_hat], 'Ac1_hats': [Ac1_hat],'A2_hats': [A2_hat], 'Ac2_hats': [Ac2_hat] }
mu=2
nu0=1
all_data={}

for seed in seeds:
    np.random.seed(seed)
    x0=np.random.rand(2,d)
    all_data[seed]={}
    all_data[seed]['x0']=x0

    x_sgd=[x0]
    x_agd=[x0]
    x_rgd=[x0]
    x_rr =[np.zeros((2,d))]
    epsilon_1 = 0
    epsilon_2 = 0
    gamma = 2.1
    alpha = 0.1

    for i in range(MAXITER):
        nu=2*nu0/(len(x_agd)+2*3*d)
        th=1*np.random.normal(0,sigma_theta,size=(d,m))
        z1=ddg.D_w(0)
        z2=ddg.D_w(1)
        x_sgd.append(ddg.proj(x_sgd[-1]-eta*ddg.getgrad(x_sgd[-1],th)))

        x_agd.append(ddg.proj(x_agd[-1]-eta*ddg.getgrad_agd(x_agd[-1],th,A1hat=A_dic['A1_hats'][-1],Ac1hat=A_dic['Ac1_hats'][-1],
                                                            A2hat=A_dic['A2_hats'][-1], Ac2hat=A_dic['Ac2_hats'][-1], passvals=True)))
        A1_hat,Ac1_hat,A2_hat,Ac2_hat = ddg.update_estimate(x_agd[-1], z1, z2,th,nu=nu,mu=mu, A1hat=A_dic['A1_hats'][-1],Ac1hat=A_dic['Ac1_hats'][-1],
                                                            A2hat=A_dic['A2_hats'][-1], Ac2hat=A_dic['Ac2_hats'][-1], passvals=True,UNCORR=False)

        A_dic['A1_hats'].append(A1_hat)
        A_dic['Ac1_hats'].append(Ac1_hat)
        A_dic['A2_hats'].append(A2_hat)
        A_dic['Ac2_hats'].append(Ac2_hat)

        z1,z2 = ddg.distribution_map(x_rgd[-1],th)
        x_rgd.append(ddg.proj(x_rgd[-1]-eta*ddg.getgrad_rgd(x_rgd[-1],z1,z2, th)))

        # repeat retraining
        z1_t_1,z2_t_1 = ddg.distribution_map(x_rr[-1],th)
        rr_model_1 = Ridge(alpha = alpha)
        rr_model_1.fit(th.T,z1_t_1,sample_weight=1/m)
        rr_coef_1 = rr_model_1.coef_
        rr_model_2 = Ridge(alpha = alpha)
        rr_model_2.fit(th.T,z2_t_1,sample_weight=1/m)
        rr_coef_2 = rr_model_2.coef_
        x_rr.append(np.vstack((rr_model_1.coef_,rr_model_2.coef_)))

        z1_t,z2_t = ddg.distribution_map(x_rr[-1],th)
        g1_t=-th@(z1_t-th.T@x_rr[-1][0])
        g2_t=-th@(z2_t-th.T@x_rr[-1][1])
        g1_t_1=-th@(z1_t_1-th.T@x_rr[-1][0])
        g2_t_1=-th@(z2_t_1-th.T@x_rr[-1][1])
        epsilon_1 = max(epsilon_1,la.norm(g1_t-g1_t_1)/la.norm(x_rr[-1][0]-x_rr[-2][0]))
        epsilon_2 = max(epsilon_2,la.norm(g2_t-g2_t_1)/la.norm(x_rr[-1][0]-x_rr[-2][0]))
        if la.norm(x_rr[-1][0]-x_rr[-2][0]) > 1e-3 or la.norm(x_rr[-1][1]-x_rr[-2][1]) > 1e-3:
            alpha = gamma*((epsilon_1**2+epsilon_2**2)**0.5)
            
    x_sgd=np.asarray(x_sgd)
    x_agd=np.asarray(x_agd)
    x_rgd=np.asarray(x_rgd)
    x_rr=np.asarray(x_rr)
    # nash=[]

    # for i in range(n):
    #     nash.append(np.mean(x_sgd[-100:,i,:],axis=0)) #np.mean(x_rgd_ps[-1000:,:,i],axis=0
    # nash=np.asarray(nash)
    # print(nash)

    error_sgd=[]
    error_agd=[]
    error_rgd=[]
    error_rr=[]

    # estimate the loss
    th=1*np.random.normal(0,sigma_theta,size=(d,m))
    for x,y,z,rr in zip(x_sgd,x_agd,x_rgd,x_rr):
        z1,z2 = ddg.distribution_map(x,th)
        error_sgd.append((la.norm(z1-th.T@x[0])**2 + la.norm(z2-th.T@x[1])**2)/m)

        z1,z2 = ddg.distribution_map(y,th)
        error_agd.append((la.norm(z1-th.T@y[0])**2 + la.norm(z2-th.T@y[1])**2)/m)

        z1,z2 = ddg.distribution_map(z,th)
        error_rgd.append((la.norm(z1-th.T@z[0])**2 + la.norm(z2-th.T@z[1])**2)/m)

        z1,z2 = ddg.distribution_map(rr,th)
        error_rr.append((la.norm(z1-th.T@rr[0])**2 + la.norm(z2-th.T@rr[1])**2)/m)

    err_agd=np.asarray(error_agd)
    err_sgd=np.asarray(error_sgd)#
    err_rgd=np.asarray(error_rgd)
    err_rr=np.asarray(error_rr)

    all_data[seed]['error_agd']=err_agd
    all_data[seed]['error_sgd']=err_sgd
    all_data[seed]['error_rgd']=err_rgd
    all_data[seed]['error_rr']=err_rr

file_name_npy = 'aaa.npz'
np.savez(file_name_npy, all_data=all_data)
print(f"Data saved to {file_name_npy}")

## Generate Plots
filename='./figs/vector_sp_theta_'+str(sigma_theta)+'_density_'+str(density)+'_mu_A_'+str(mu_A)+'_mu_AC_'+str(mu_AC)+'.'
print(f"Figure plot to {filename}")
SAVE=1

errs_agd=[]
errs_sgd=[]
errs_dfo=[]
errs_rgd=[]
errs_rr = []
fs=24
for seed in seeds:
    errs_agd.append(all_data[seed]['error_agd'])
    errs_sgd.append(all_data[seed]['error_sgd'])
    errs_rgd.append(all_data[seed]['error_rgd'])
    errs_rr.append(all_data[seed]['error_rr'])

errs_agd=np.asarray(errs_agd)
errs_sgd=np.asarray(errs_sgd)
errs_rgd=np.asarray(errs_rgd)
errs_rr=np.asarray(errs_rr)


errs_agd_mean=np.mean(errs_agd,axis=0)
errs_sgd_mean=np.mean(errs_sgd,axis=0)
errs_rgd_mean=np.mean(errs_rgd,axis=0)
errs_rr_mean=np.mean(errs_rr,axis=0)

errs_agd_var=np.std(errs_agd,axis=0)
errs_sgd_var=np.std(errs_sgd,axis=0)
errs_rgd_var=np.std(errs_rgd,axis=0)
errs_rr_var=np.std(errs_rr,axis=0)
# print(np.shape(errs_agd_var))

iterations=np.arange(0,MAXITER+1)
fig=plt.figure(figsize=(10,7))
# for i in range(len(errs_agd)):
#     # plt.plot(errs_sgd[i,:], linewidth=3,color='xkcd:cerulean', alpha=0.1)
#     plt.plot(errs_agd[i,:], linewidth=3, alpha=0.1, color='xkcd:light orange')
#     plt.plot(errs_rgd[i,:], linewidth=3, alpha=0.1, color='xkcd:light green')
    

plt.plot(errs_sgd_mean, linewidth=3,color='xkcd:cerulean', label='SGM')
plt.fill_between(iterations,errs_sgd_mean+errs_sgd_var,errs_sgd_mean-errs_sgd_var, alpha=0.5, linewidth=0,color='xkcd:cerulean')
plt.plot(errs_agd_mean, linewidth=3, color='xkcd:light orange', label='AGM')
plt.plot(errs_rgd_mean, linewidth=3, color='xkcd:light green', label='RGD')
plt.plot(errs_rr_mean, linewidth=3, color='xkcd:light blue', label='Ours_RR')
plt.fill_between(iterations,errs_agd_mean+errs_agd_var,errs_agd_mean-errs_agd_var, alpha=0.4, linewidth=0, color='xkcd:light orange')
plt.fill_between(iterations,errs_rgd_mean+errs_rgd_var,errs_rgd_mean-errs_rgd_var, alpha=0.4, linewidth=0, color='xkcd:light green')
plt.fill_between(iterations,errs_rr_mean+errs_rr_var,errs_rr_mean-errs_rr_var, alpha=0.4, linewidth=0, color='xkcd:light blue')
plt.yscale('log')
plt.grid(True)

plt.tick_params(labelsize=fs-2)
# plt.ylabel(r'$\mathbb{E}\sum_{i=1}^n \Vert z_i^t- \theta^\top x_i^t\Vert^2$', fontsize=fs)
plt.ylabel(r'MSE', fontsize=fs)
plt.xlabel(r'iterations', fontsize=fs)
plt.legend(fontsize=fs-2, loc='upper right',ncol=2)
if SAVE:
    for tag in ['pdf']:
        plt.savefig(filename+tag,  bbox_inches='tight', dpi=300)
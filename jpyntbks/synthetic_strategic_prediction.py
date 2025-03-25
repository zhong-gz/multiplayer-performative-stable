import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
import seaborn as sns
import random
import sys, os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilssp import *
# %load_ext autoreload
# %autoreload 2

seeds=random.sample(range(1000), 50)
N=1000
N_test=100
MAXITER=100
sigma_theta=0.01
sigma_w=0.01
sigma_z1=1.1
sigma_z2=1.1
density=0.5
m=10 # both players dimension of z_i
d=2 # size of each players action
B=np.ones((d,1)) #np.array([[1],[1]]) #np.random.rand(d,1)
np.random.seed(2)
A1=1*scirand(m,d,density=density).A
Ac1=0.5*scirand(m,d,density=density).A
A2=1*scirand(m,d,density=density).A
Ac2=0.5*scirand(m,d,density=density).A
params={'A1':A1,'A2':A2,'Ac1':Ac1,'Ac2':Ac2} 

# lam=[1.0,1.0]
# lam=[0.1,0.1]
lam=[0,0]
MAXITER=1000
n=2
ddg=ddstrategic_prediction(MAXITER=MAXITER, sigma_theta=0.01,sigma_w=0.01,sigma_z1=0.01,sigma_z2=0.01,density=density,
                       B=B,nu=1e-3, lam=lam,N_test=N_test,n=n, m=m, d=d, params=params,
                           mu_w1=0, mu_w2=0, mu_theta=0)

_,S1,_=la.svd(A1)
_,S2,_=la.svd(A2)
S1=np.sort(S1)[-1]
S2=np.sort(S2)[-1]
eta=0.01 #1/np.max([S1,S2])
eta_agd=0.01 #1/np.max([S1,S2])
nu=1e-2

A1_hat = np.zeros((m,d)) #np.random.rand(m,d)
Ac1_hat =np.zeros((m,d))# np.random.rand(m,d)
A2_hat = np.zeros((m,d)) #np.random.rand(m,d)
Ac2_hat = np.zeros((m,d)) #np.random.rand(m,d)

A_dic={'A1_hats':[A1_hat], 'Ac1_hats': [Ac1_hat],'A2_hats': [A2_hat], 'Ac2_hats': [Ac2_hat] }
mu=2
nu0=1
eta_agd=0.01 #5e-2 #1e-4
all_data={}

for seed in seeds:
    np.random.seed(seed)
    x0=np.random.rand(2,d)
    all_data[seed]={}
    all_data[seed]['x0']=x0

    x_sgd=[x0]
    x_agd=[x0]
    x_rgd=[x0]

    for i in range(MAXITER):
        nu=2*nu0/(len(x_agd)+2*3*d)
        th=1*np.random.normal(0,sigma_theta,size=(d,m))
        z1=ddg.D_w(0)
        z2=ddg.D_w(1)
        x_sgd.append(ddg.proj(x_sgd[-1]-eta*ddg.getgrad(x_sgd[-1],th)))

        x_agd.append(ddg.proj(x_agd[-1]-eta_agd*ddg.getgrad_agd(x_agd[-1],th,A1hat=A_dic['A1_hats'][-1],Ac1hat=A_dic['Ac1_hats'][-1],
                                                            A2hat=A_dic['A2_hats'][-1], Ac2hat=A_dic['Ac2_hats'][-1], passvals=True)))
        A1_hat,Ac1_hat,A2_hat,Ac2_hat = ddg.update_estimate(x_agd[-1], z1, z2,th,nu=nu,mu=mu, A1hat=A_dic['A1_hats'][-1],Ac1hat=A_dic['Ac1_hats'][-1],
                                                            A2hat=A_dic['A2_hats'][-1], Ac2hat=A_dic['Ac2_hats'][-1], passvals=True,UNCORR=False)
        # print('x_agd:',x_agd[-1])

        A_dic['A1_hats'].append(A1_hat)
        A_dic['Ac1_hats'].append(Ac1_hat)
        A_dic['A2_hats'].append(A2_hat)
        A_dic['Ac2_hats'].append(Ac2_hat)

        z1,z2 = ddg.distribution_map(x_rgd[-1],th)
        x_rgd.append(ddg.proj(x_rgd[-1]-eta*ddg.getgrad_rgd(x_rgd[-1],z1,z2, th)))

    x_sgd=np.asarray(x_sgd)
    x_agd=np.asarray(x_agd)
    x_rgd=np.asarray(x_rgd)
    # nash=[]

    # for i in range(n):
    #     nash.append(np.mean(x_sgd[-100:,i,:],axis=0)) #np.mean(x_rgd_ps[-1000:,:,i],axis=0
    # nash=np.asarray(nash)
    # print(nash)

    error_sgd=[]
    error_agd=[]
    error_rgd=[]

    # estimate the loss
    th=1*np.random.normal(0,sigma_theta,size=(d,m))
    for x,y,z in zip(x_sgd,x_agd,x_rgd):
        z1,z2 = ddg.distribution_map(x,th)
        error_sgd.append(la.norm(z1-th.T@x[0])**2 + la.norm(z2-th.T@x[1])**2)

        z1,z2 = ddg.distribution_map(y,th)
        error_agd.append(la.norm(z1-th.T@y[0])**2 + la.norm(z2-th.T@y[1])**2)

        z1,z2 = ddg.distribution_map(z,th)
        error_rgd.append(la.norm(z1-th.T@z[0])**2 + la.norm(z2-th.T@z[1])**2)

    err_agd=np.asarray(error_agd)
    err_sgd=np.asarray(error_sgd)#
    err_rgd=np.asarray(error_rgd)

    all_data[seed]['error_agd']=err_agd
    all_data[seed]['error_sgd']=err_sgd
    all_data[seed]['error_rgd']=err_rgd

## Generate Plots
filename='./figs/convergence_final.'
SAVE=1

errs_agd=[]
errs_sgd=[]
errs_dfo=[]
errs_rgd=[]
fs=24
for seed in seeds:
    errs_agd.append(all_data[seed]['error_agd'])
    errs_sgd.append(all_data[seed]['error_sgd'])
    errs_rgd.append(all_data[seed]['error_rgd'])

errs_agd=np.asarray(errs_agd)
errs_sgd=np.asarray(errs_sgd)
errs_rgd=np.asarray(errs_rgd)


errs_agd_mean=np.mean(errs_agd,axis=0)
errs_sgd_mean=np.mean(errs_sgd,axis=0)
errs_rgd_mean=np.mean(errs_rgd,axis=0)

errs_agd_var=np.std(errs_agd,axis=0)
errs_sgd_var=np.std(errs_sgd,axis=0)
errs_rgd_var=np.std(errs_rgd,axis=0)
# print(np.shape(errs_agd_var))

iterations=np.arange(0,MAXITER+1)
fig=plt.figure(figsize=(10,7))
for i in range(len(errs_agd)):
    # plt.plot(errs_sgd[i,:], linewidth=3,color='xkcd:cerulean', alpha=0.1)
    plt.plot(errs_agd[i,:], linewidth=3, alpha=0.1, color='xkcd:light orange')
    plt.plot(errs_rgd[i,:], linewidth=3, alpha=0.1, color='xkcd:light green')
    

# plt.plot(errs_sgd_mean, linewidth=3,color='xkcd:cerulean', label='SGM')
# plt.fill_between(iterations,errs_sgd_mean+errs_sgd_var,errs_sgd_mean-errs_sgd_var, alpha=0.5, linewidth=0,color='xkcd:cerulean')
plt.plot(errs_agd_mean, linewidth=3, color='xkcd:light orange', label='AGM')
plt.yscale('log')
plt.fill_between(iterations,errs_agd_mean+errs_agd_var,errs_agd_mean-errs_agd_var, alpha=0.5, linewidth=0, color='xkcd:light orange')
plt.plot(errs_rgd_mean, linewidth=3, color='xkcd:light green', label='RGD')
plt.fill_between(iterations,errs_rgd_mean+errs_rgd_var,errs_rgd_mean-errs_rgd_var, alpha=0.5, linewidth=0, color='xkcd:light green')
plt.grid(True)

plt.tick_params(labelsize=fs-2)
plt.ylabel(r'$\Vert x^t-x^\ast\Vert^2$', fontsize=fs)
plt.xlabel(r'iterations', fontsize=fs)
plt.legend(fontsize=fs-2, loc='lower left',ncol=2)
if SAVE:
    for tag in ['pdf']:
        plt.savefig(filename+tag,  bbox_inches='tight', dpi=300)
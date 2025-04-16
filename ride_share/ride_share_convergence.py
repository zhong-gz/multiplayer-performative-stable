import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import pickle 
import sys, os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./utils/' )
from utilsrm import *
import scipy.linalg  as sla
import random

import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
# %load_ext autoreload
# %autoreload 2
# https://plotly.com/python/static-image-export/ need to install this if you want to save images
import plotly.express as px
px.set_mapbox_access_token("pk.eyJ1IjoicmF0bGlmZmxqIiwiYSI6ImNqOGJ4cm8wcjAzN3QyeG1zcnZvMjB5bGUifQ.iRkpBPE-WANBkVc9ffI8ng")

## Initialize the game class and set the random seed and initial point
seed = 42
num_experiments = 5
np.random.seed(seed)

## Experiment 1: Convergence

BATCH=20
loc_cap=11
nu=0.0001 #0.001 and B=4 #0.0005 B=5 #0.00025 B=5/6
eta= 5e-5 #1e-4 0.001 
lam1=1
lam2=1

loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
x0=np.random.rand(2,loc_cap)

## rr
gamma = 2.1

MAXITER=3000
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=seed,lam=[lam1,lam2], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()
seeds=random.sample(range(100), num_experiments)

all_data={}
for seed in seeds:
    x0=np.random.rand(2,loc_cap)
    all_data[seed]={}
    all_data[seed]['x0']=x0
    x_agd=[x0]
    x_sgd=[x0]
    A1_hat = -10*np.random.rand(np.shape(ddgame.A1)[1])
    Ac1_hat = 2*np.random.rand(np.shape(ddgame.Ac1)[1])
    A2_hat = -10*np.random.rand(np.shape(ddgame.A2)[1])
    Ac2_hat = 2*np.random.rand(np.shape(ddgame.Ac2)[1])
    A1_hat=np.diag(A1_hat)
    Ac1_hat=np.diag(Ac1_hat)
    A2_hat=np.diag(A2_hat)
    Ac2_hat=np.diag(Ac2_hat)
    A_dic={}
    A_dic['A1_hat']=A1_hat
    A_dic['Ac1_hat']=Ac1_hat
    A_dic['A2_hat']=A2_hat
    A_dic['Ac2_hat']=Ac2_hat

    dic_agd=ddgame.runAGD(x0,A_dic,eta=eta,nu=nu,BATCH=BATCH,MAXITER=MAXITER, perform_agd=[True,True], INNERITER=1, B=6,UNCORR=True) #inner was 100
    # dic_sgd=ddgame.runSGD(x0,eta=eta,BATCH=BATCH,MAXITER=MAXITER, perform_sgd=[True,True])
    dic_rr = ddgame.runRR(gamma = gamma,BATCH=BATCH,MAXITER=MAXITER, perform_rr=[True,True])
        
    x_agd=np.asarray(dic_agd['x'])
    # x_sgd=np.asarray(dic_sgd['x'])
    x_rr =np.asarray(dic_rr['x'])

    # error_sgd=[]
    # error_agd=[]
    # for y,agd in zip(x_sgd,x_agd):
    #     # ddgame.loss(agd,0,q_lyft_)
    #     error_sgd.append(la.norm(y)**2)
    #     error_agd.append(la.norm(agd)**2)
    
    error_agd = dic_agd['revenue_total_p1']+dic_agd['revenue_total_p2']
    # error_sgd = dic_sgd['revenue_total_p1']+dic_sgd['revenue_total_p2']
    error_rr = dic_rr['revenue_total_p1']+dic_rr['revenue_total_p2']

    # err_sgd=np.asarray(error_sgd)
    err_agd=np.asarray(error_agd)
    err_rr=np.asarray(error_rr)
    all_data[seed]['error_agd']=err_agd
    # all_data[seed]['error_sgd']=err_sgd
    all_data[seed]['error_rr']=err_rr

filename='./figs_ride_share/convergence_rideshare_DFO_loc11.'
errs_agd=[]
# errs_sgd=[]
errs_rr=[]
fs=24
for seed in seeds:
    errs_agd.append(all_data[seed]['error_agd'])
    # errs_sgd.append(all_data[seed]['error_sgd'])
    errs_rr.append(all_data[seed]['error_rr'])
errs_agd=np.asarray(errs_agd)
# errs_sgd=np.asarray(errs_sgd)
errs_rr=np.asarray(errs_rr)

errs_agd_mean=np.mean(errs_agd,axis=0)
# errs_sgd_mean=np.mean(errs_sgd,axis=0)
errs_rr_mean=np.mean(errs_rr,axis=0)

errs_agd_var=np.std(errs_agd,axis=0)
# errs_sgd_var=np.std(errs_sgd,axis=0)
errs_rr_var=np.std(errs_rr,axis=0)
# print(np.shape(errs_agd_var))
iterations=np.arange(0,MAXITER+1)
fig=plt.figure(figsize=(10,7))
for i in range(len(errs_agd)):
    plt.plot(errs_agd[i,:], linewidth=3, alpha=0.1, color='xkcd:light orange')
    # plt.plot(errs_sgd[i,:], linewidth=3,color='xkcd:cerulean', alpha=0.1)
    plt.plot(errs_rr[i,:], linewidth=3,color='xkcd:light green', alpha=0.1)
plt.plot(errs_agd_mean, linewidth=3, color='xkcd:light orange', label='AGM')
plt.fill_between(iterations,errs_agd_mean+errs_agd_var,errs_agd_mean-errs_agd_var, alpha=0.5, linewidth=0, color='xkcd:light orange')
# plt.plot(errs_sgd_mean, linewidth=3,color='xkcd:cerulean', label='SGM')
# plt.fill_between(iterations,errs_sgd_mean+errs_sgd_var,errs_sgd_mean-errs_sgd_var, alpha=0.5, linewidth=0,color='xkcd:cerulean')
plt.plot(errs_rr_mean, linewidth=3,color='xkcd:light green', label='RR')
plt.fill_between(iterations,errs_rr_mean+errs_rr_var,errs_rr_mean-errs_rr_var, alpha=0.5, linewidth=0,color='xkcd:light green')
plt.tick_params(labelsize=fs-2)
# plt.yscale('log')
plt.grid(True)
plt.ylabel(r'Total Revenue', fontsize=fs)
plt.xlabel(r'iterations', fontsize=fs)
plt.legend(fontsize=fs-2, loc='lower left',ncol=1)

for tag in ['pdf']:
    plt.savefig(filename+tag,  bbox_inches='tight', dpi=300)


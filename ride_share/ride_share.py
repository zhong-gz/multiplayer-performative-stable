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

# set flags
verbose=False # print out stuff for debugging
centered=False # centering the data

loc_cap=11
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()

verbose=False
BATCH=10
MAXITER=5000
np.random.seed(10)
eta=0.001 
x0=np.random.rand(2,loc_cap)

def zograd(x, z1_, z2_, A_1=ddgame.A1, A_1_=ddgame.Ac1, A_2=ddgame.A2, A_2_=ddgame.Ac2, delta=0.001, BATCH=1):
    p1 = np.zeros(ddgame.d)
    p2 = np.zeros(ddgame.d)
    for sample in range(BATCH):
        v1_ = np.random.normal(0,1,size=(ddgame.d,))
        v1 = v1_/la.norm(v1_)
        v2_ = np.random.normal(0,1,size=(ddgame.d,))
        v2 = v2_/la.norm(v2_)

        z1 = z1_+A_1@(x[0]+(delta*v1))+A_1_@(x[1]+(delta*v2))
        l1 = -0.5*z1@(x[0]+(delta*v1))+ddgame.lam1*la.norm(x[0]+(delta*v1))
        p1 += (ddgame.d/delta)*l1*v1

        z2 = z2_+A_2@(x[1]+(delta*v2))+A_2_@(x[0]+(delta*v1))
        l2 = -0.5*z2@(x[1]+(delta*v2))+ddgame.lam2*la.norm(x[1]+(delta*v2))
        p2 += (ddgame.d/delta)*l2*v2
        
    p1=p1/BATCH
    p2=p2/BATCH
    return np.vstack((p1,p2))

verbose=False
BATCH=20


loc_cap=11
nu=0.0001 #0.001 and B=4 #0.0005 B=5 #0.00025 B=5/6
eta= 5e-5 #1e-4
lam1=2
lam2=2
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
x0=np.random.rand(2,loc_cap)
# computes Nash
MAXITER=50000

loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[lam1,lam2], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()
dic_sgd=ddgame.runSGD(x0,eta=0.001,BATCH=10,MAXITER=MAXITER, perform_sgd=[True,True])
x_sgd=np.asarray(dic_sgd['x'])

nash=[]
for i in range(loc_cap):
    nash.append(np.mean(x_sgd[-100:,:,i],axis=0))
nash=np.asarray(nash)

MAXITER=5000
eta=5e-5
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[lam1,lam2], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()
seeds=random.sample(range(1000), 2)

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
    dic_sgd=ddgame.runSGD(x0,eta=eta,BATCH=BATCH,MAXITER=MAXITER, perform_sgd=[True,True])
    x_agd=np.asarray(dic_agd['x'])
    x_sgd=np.asarray(dic_sgd['x'])

    alpha = np.min([la.norm(-ddgame.A1+ddgame.lam1*np.eye(ddgame.d)),la.norm(-ddgame.A2+ddgame.lam2*np.eye(ddgame.d))])-0.5*la.norm(ddgame.Ac1+ddgame.Ac2.T)

    eta_zo = 2
    delta = 10*0.5

    x_dfo=[x0]
    z1_base=ddgame.ql_[:,:,0].T
    z2_base=ddgame.qu_[:,:,0].T
    rev_dfo_p1=[ddgame.revenue(x_dfo[-1],0,z1_base)]
    rev_dfo_p2=[ddgame.revenue(x_dfo[-1],1,z2_base)]
    for i in range(MAXITER):
        z1_=ddgame.D_z(z1_base, batch=BATCH)
        z2_=ddgame.D_z(z2_base, batch=BATCH)


        x_dfo.append(ddgame.proj(x_dfo[-1]-(eta_zo/(alpha*(i+1))*zograd(x_dfo[-1], z1_, z2_, ddgame.A1, ddgame.Ac1, ddgame.A2, ddgame.Ac2, delta, BATCH=50))))
        rev_dfo_p1.append(ddgame.revenue(x_dfo[-1],0,z1_base))
        rev_dfo_p2.append(ddgame.revenue(x_dfo[-1],1,z2_base))

    x_dfo=np.asarray(x_dfo)

    error_dfo=[]
    error_sgd=[]
    error_agd=[]
    for x,y in zip(x_dfo,x_sgd):
        error_dfo.append(la.norm(x-nash.T)**2)
        error_sgd.append(la.norm(y-nash.T)**2)

    err_dfo=np.asarray(error_dfo)
    err_sgd=np.asarray(error_sgd)

    for x in x_agd:
        error_agd.append(la.norm(x-nash.T)**2)

    err_agd=np.asarray(error_agd)
    all_data[seed]['error_agd']=err_agd
    all_data[seed]['error_sgd']=err_sgd
    all_data[seed]['error_dfo']=err_dfo

filename='./figs_rider_share/convergence_rideshare_DFO_loc11.'
errs_agd=[]
errs_sgd=[]
errs_dfo=[]
fs=24
for seed in seeds:
    errs_agd.append(all_data[seed]['error_agd'])
    errs_sgd.append(all_data[seed]['error_sgd'])
    errs_dfo.append(all_data[seed]['error_dfo'])
errs_agd=np.asarray(errs_agd)
errs_sgd=np.asarray(errs_sgd)
errs_dfo=np.asarray(errs_dfo)


errs_agd_mean=np.mean(errs_agd,axis=0)
errs_sgd_mean=np.mean(errs_sgd,axis=0)
errs_dfo_mean=np.mean(errs_dfo,axis=0)

errs_agd_var=np.std(errs_agd,axis=0)
errs_sgd_var=np.std(errs_sgd,axis=0)
# print(np.shape(errs_agd_var))
errs_dfo_var=np.std(errs_dfo,axis=0)
iterations=np.arange(0,MAXITER+1)
fig=plt.figure(figsize=(10,7))
for i in range(len(errs_agd)):
    plt.plot(errs_agd[i,:], linewidth=3, alpha=0.1, color='xkcd:light orange')
    plt.plot(errs_sgd[i,:], linewidth=3,color='xkcd:cerulean', alpha=0.1)
    plt.plot(errs_dfo[i,:], linewidth=3,color='xkcd:tomato red',alpha=0.1)
plt.plot(errs_agd_mean, linewidth=3, color='xkcd:light orange', label='AGM')
plt.yscale('log')
plt.fill_between(iterations,errs_agd_mean+errs_agd_var,errs_agd_mean-errs_agd_var, alpha=0.5, linewidth=0, color='xkcd:light orange')
plt.grid(True)
plt.plot(errs_sgd_mean, linewidth=3,color='xkcd:cerulean', label='SGM')
plt.fill_between(iterations,errs_sgd_mean+errs_sgd_var,errs_sgd_mean-errs_sgd_var, alpha=0.5, linewidth=0,color='xkcd:cerulean')
plt.plot(errs_dfo_mean, linewidth=3,color='xkcd:tomato red', label='DFM')
plt.fill_between(iterations,errs_dfo_mean+errs_dfo_var,errs_dfo_mean-errs_dfo_var, alpha=0.5, linewidth=0,color='xkcd:tomato red',)
plt.tick_params(labelsize=fs-2)
plt.ylabel(r'$\Vert x^t-x^\ast\Vert^2$', fontsize=fs)
plt.xlabel(r'iterations', fontsize=fs)
plt.legend(fontsize=fs-2, loc='upper right',ncol=3)

for tag in ['pdf']:
    plt.savefig(filename+tag,  bbox_inches='tight', dpi=300)


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
filepath = './figs_ride_share/'

## Experiment 1: Convergence

BATCH=20
loc_cap=11
nu=0.0001 #0.001 and B=4 #0.0005 B=5 #0.00025 B=5/6
eta= 0.0001 #5e-5 #1e-4 0.001 
lam1= 10
lam2= 10
for p in [0,1,2,3,4]:  #
    price_index = p

    loc_lst_index=list(range(0,loc_cap))
    price_lst_index=list(range(0,5))
    x0=np.random.rand(2,loc_cap)

    ## rr
    gamma = 2.1

    MAXITER=2000
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

        dic_agd=ddgame.runAGD(x0,A_dic,price_index=price_index,eta=eta,nu=nu,BATCH=BATCH,MAXITER=MAXITER, perform_agd=[True,True], INNERITER=1, B=6,UNCORR=True) #inner was 100
        # dic_sgd=ddgame.runSGD(x0,eta=eta,BATCH=BATCH,MAXITER=MAXITER, perform_sgd=[True,True])
        dic_rgd = ddgame.runRGD(x0,price_index=price_index,eta=eta,BATCH=BATCH,MAXITER=MAXITER) 
        dic_sfb = ddgame.runSFB(x0,price_index=price_index,eta=eta,BATCH=BATCH,MAXITER=MAXITER) 
        dic_opg =ddgame.runOPGD(x0,price_index=price_index,eta=eta,BATCH=BATCH,MAXITER=MAXITER, perform_opgd=[True,True])
        dic_rr = ddgame.runRR(gamma = gamma,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER, perform_rr=[True,True])
        
        x_agd=np.asarray(dic_agd['x'])
        x_rgd=np.asarray(dic_rgd['x'])
        x_sfb=np.asarray(dic_sfb['x'])
        x_opg = np.asarray(dic_opg['x'])
        x_rr =np.asarray(dic_rr['x'])
        
        error_agd = dic_agd['revenue_total_p1']+dic_agd['revenue_total_p2']
        error_rgd = dic_rgd['revenue_total_p1']+dic_rgd['revenue_total_p2']
        error_sfb = dic_sfb['revenue_total_p1']+dic_sfb['revenue_total_p2']
        error_opg = dic_opg['revenue_total_p1']+dic_opg['revenue_total_p2']
        error_rr =  dic_rr['revenue_total_p1'] +dic_rr['revenue_total_p2']

        rev_lyft_agd = np.asarray(dic_agd['revenue_total_p1'])
        rev_uber_avd = np.asarray(dic_agd['revenue_total_p2'])
        rev_lyft_rgd = np.asarray(dic_rgd['revenue_total_p1'])
        rev_uber_rgd = np.asarray(dic_rgd['revenue_total_p2'])
        rev_lyft_sfb = np.asarray(dic_sfb['revenue_total_p1'])
        rev_uber_sfb = np.asarray(dic_sfb['revenue_total_p2'])
        rev_lyft_opg = np.asarray(dic_opg['revenue_total_p1'])
        rev_uber_opg = np.asarray(dic_opg['revenue_total_p2'])
        rev_lyft_rr =  np.asarray(dic_rr['revenue_total_p1'])
        rev_uber_rr =  np.asarray(dic_rr['revenue_total_p2'])

        err_agd=np.asarray(error_agd)
        err_rgd=np.asarray(error_rgd)
        err_sfb=np.asarray(error_sfb)
        err_opg=np.asarray(error_opg)
        err_rr=np.asarray(error_rr)
        all_data[seed]['error_agd']=err_agd
        all_data[seed]['error_rgd']=err_rgd
        all_data[seed]['error_sfb']=err_sfb
        all_data[seed]['error_opg']=err_opg
        all_data[seed]['error_rr']=err_rr
        all_data[seed]['rev_lyft_agd']=rev_lyft_agd
        all_data[seed]['rev_uber_agd']=rev_uber_avd
        all_data[seed]['rev_lyft_rgd']=rev_lyft_rgd
        all_data[seed]['rev_uber_rgd']=rev_uber_rgd
        all_data[seed]['rev_lyft_sfb']=rev_lyft_sfb
        all_data[seed]['rev_uber_sfb']=rev_uber_sfb
        all_data[seed]['rev_lyft_opg']=rev_lyft_opg
        all_data[seed]['rev_uber_opg']=rev_uber_opg
        all_data[seed]['rev_lyft_rr']=rev_lyft_rr
        all_data[seed]['rev_uber_rr']=rev_uber_rr

    filename= filepath+str((price_index*5+10))+'_convergence_rideshare_total_revenue.'
    errs_agd=[]
    errs_rgd=[]
    errs_sfb=[]
    errs_opg=[]
    errs_rr=[]
    revs_lyft_agd = []
    revs_uber_agd = []
    revs_lyft_rgd = []
    revs_uber_rgd = []
    revs_lyft_sfb = []
    revs_uber_sfb = []
    revs_lyft_opg = []
    revs_uber_opg = []
    revs_lyft_rr = []
    revs_uber_rr = []
    fs=24
    for seed in seeds:
        errs_agd.append(all_data[seed]['error_agd'])
        errs_rgd.append(all_data[seed]['error_rgd'])
        errs_sfb.append(all_data[seed]['error_sfb'])
        errs_opg.append(all_data[seed]['error_opg'])
        errs_rr.append(all_data[seed]['error_rr'])
        revs_lyft_agd.append(all_data[seed]['rev_lyft_agd'])
        revs_uber_agd.append(all_data[seed]['rev_uber_agd'])
        revs_lyft_rgd.append(all_data[seed]['rev_lyft_rgd'])
        revs_uber_rgd.append(all_data[seed]['rev_uber_rgd'])
        revs_lyft_sfb.append(all_data[seed]['rev_lyft_sfb'])
        revs_uber_sfb.append(all_data[seed]['rev_uber_sfb'])
        revs_lyft_opg.append(all_data[seed]['rev_lyft_opg'])
        revs_uber_opg.append(all_data[seed]['rev_uber_opg'])
        revs_lyft_rr.append(all_data[seed]['rev_lyft_rr'])
        revs_uber_rr.append(all_data[seed]['rev_uber_rr'])
    errs_agd=np.asarray(errs_agd)
    errs_rgd=np.asarray(errs_rgd)
    errs_sfb=np.asarray(errs_sfb)
    errs_opg=np.asarray(errs_opg)
    errs_rr=np.asarray(errs_rr)
    revs_lyft_agd = np.asarray(revs_lyft_agd)
    revs_uber_agd = np.asarray(revs_uber_agd)
    revs_lyft_rgd = np.asarray(revs_lyft_rgd)
    revs_uber_rgd = np.asarray(revs_uber_rgd)
    revs_lyft_sfb = np.asarray(revs_lyft_sfb)
    revs_uber_sfb = np.asarray(revs_uber_sfb)
    revs_lyft_opg = np.asarray(revs_lyft_opg)
    revs_uber_opg = np.asarray(revs_uber_opg)
    revs_lyft_rr = np.asarray(revs_lyft_rr)
    revs_uber_rr = np.asarray(revs_uber_rr)

    errs_agd_mean=np.mean(errs_agd,axis=0)
    errs_rgd_mean=np.mean(errs_rgd,axis=0)
    errs_sfb_mean=np.mean(errs_sfb,axis=0)
    errs_opg_mean=np.mean(errs_opg,axis=0)
    errs_rr_mean=np.mean(errs_rr,axis=0)
    revs_lyft_agd_mean = np.mean(revs_lyft_agd,axis=0)
    revs_uber_agd_mean = np.mean(revs_uber_agd,axis=0)
    revs_lyft_rgd_mean = np.mean(revs_lyft_rgd,axis=0)
    revs_uber_rgd_mean = np.mean(revs_uber_rgd,axis=0)
    revs_lyft_sfb_mean = np.mean(revs_lyft_sfb,axis=0)
    revs_uber_sfb_mean = np.mean(revs_uber_sfb,axis=0)
    revs_lyft_opg_mean = np.mean(revs_lyft_opg,axis=0)
    revs_uber_opg_mean = np.mean(revs_uber_opg,axis=0)
    revs_lyft_rr_mean = np.mean(revs_lyft_rr,axis=0)
    revs_uber_rr_mean = np.mean(revs_uber_rr,axis=0)

    errs_agd_var=np.std(errs_agd,axis=0)
    errs_rgd_var=np.std(errs_rgd,axis=0)
    errs_sfb_var=np.std(errs_sfb,axis=0)
    errs_opg_var=np.std(errs_opg,axis=0)
    errs_rr_var=np.std(errs_rr,axis=0)
    revs_lyft_agd_var = np.std(revs_lyft_agd,axis=0)
    revs_uber_agd_var = np.std(revs_uber_agd,axis=0)
    revs_lyft_rgd_var = np.std(revs_lyft_rgd,axis=0)
    revs_uber_rgd_var = np.std(revs_uber_rgd,axis=0)
    revs_lyft_sfb_var = np.std(revs_lyft_sfb,axis=0)
    revs_uber_sfb_var = np.std(revs_uber_sfb,axis=0)
    revs_lyft_opg_var = np.std(revs_lyft_opg,axis=0)
    revs_uber_opg_var = np.std(revs_uber_opg,axis=0)
    revs_lyft_rr_var = np.std(revs_lyft_rr,axis=0)
    revs_uber_rr_var = np.std(revs_uber_rr,axis=0)

    iterations=np.arange(0,MAXITER+1)
    fig=plt.figure(figsize=(10,7))
    # for i in range(len(errs_agd)):
    #     plt.plot(errs_rgd[i,:], linewidth=3,color='#444444', alpha=0.1)
    #     plt.plot(errs_agd[i,:], linewidth=3, alpha=0.1, color='#9467bd')
    #     plt.plot(errs_sfb[i,:], linewidth=3, alpha=0.1, color='#2ca02c')
    #     plt.plot(errs_rr[i,:], linewidth=3,color='#FF7F50', alpha=0.1)
    plt.plot(errs_rr_mean, linewidth=3.7,color='#FF7F50', label='RR')
    plt.fill_between(iterations,errs_rr_mean+errs_rr_var,errs_rr_mean-errs_rr_var, alpha=0.5, linewidth=0,color='#FF7F50')
    plt.plot(errs_rgd_mean, linewidth=3,color='#444444', label='RGD')
    plt.fill_between(iterations,errs_rgd_mean+errs_rgd_var,errs_rgd_mean-errs_rgd_var, alpha=0.5, linewidth=0,color='#444444')
    plt.plot(errs_agd_mean, linewidth=3, color='#9467bd', label='AGM')
    plt.fill_between(iterations,errs_agd_mean+errs_agd_var,errs_agd_mean-errs_agd_var, alpha=0.5, linewidth=0, color='#9467bd')
    plt.plot(errs_sfb_mean, linewidth=3, color='#2ca02c', label='SFB')
    plt.fill_between(iterations,errs_sfb_mean+errs_sfb_var,errs_sfb_mean-errs_sfb_var, alpha=0.5, linewidth=0, color='#2ca02c')
    plt.plot(errs_opg_mean, linewidth=3, color='#1f77b4', label='OPGD')
    plt.fill_between(iterations,errs_opg_mean+errs_opg_var,errs_opg_mean-errs_opg_var, alpha=0.5, linewidth=0, color='#1f77b4')
    plt.plot(errs_rr_mean, linewidth=3.7,color='#FF7F50')

    plt.tick_params(labelsize=fs-2)
    # plt.yscale('log')
    plt.grid(True)
    plt.ylabel(r'Total Revenue', fontsize=fs)
    plt.xlabel(r'iterations', fontsize=fs)
    plt.legend(fontsize=fs-2, loc='lower right',ncol=1)
    plt.savefig(filename+'pdf',  bbox_inches='tight', dpi=300)

    filename= filepath+str((price_index*5+10))+'_convergence_rideshare_lyft_revenue.'
    fig=plt.figure(figsize=(10,7))
    plt.plot(revs_lyft_rr_mean, linewidth=3.7,color='#FF7F50', label='RR')
    plt.plot(revs_lyft_rgd_mean, linewidth=3,color='#444444', label='RGD')
    plt.plot(revs_lyft_agd_mean, linewidth=3,color='#9467bd', label='AGM')
    plt.plot(revs_lyft_sfb_mean, linewidth=3,color='#2ca02c', label='SFB')
    plt.plot(revs_lyft_opg_mean, linewidth=3,color='#1f77b4', label='OPGD')
    plt.plot(revs_lyft_rr_mean, linewidth=3.7,color='#FF7F50')
    plt.tick_params(labelsize=fs-2)
    # plt.yscale('log')
    plt.grid(True)
    plt.ylabel(r'lyft Revenue', fontsize=fs)
    plt.xlabel(r'iterations', fontsize=fs)
    plt.legend(fontsize=fs-2, loc='lower right',ncol=1)
    plt.savefig(filename+'pdf',  bbox_inches='tight', dpi=300)

    filename= filepath+str((price_index*5+10))+'_convergence_rideshare_uber_revenue.'
    fig=plt.figure(figsize=(10,7))
    plt.plot(revs_uber_rr_mean, linewidth=3.7,color='#FF7F50', label='RR')
    plt.plot(revs_uber_rgd_mean, linewidth=3,color='#444444', label='RGD')
    plt.plot(revs_uber_agd_mean, linewidth=3,color='#9467bd', label='AGM')
    plt.plot(revs_uber_sfb_mean, linewidth=3,color='#2ca02c', label='SFB')
    plt.plot(revs_uber_opg_mean, linewidth=3,color='#1f77b4', label='OPGD')
    plt.plot(revs_uber_rr_mean, linewidth=3.7,color='#FF7F50')
    plt.tick_params(labelsize=fs-2)
    # plt.yscale('log')
    plt.grid(True)
    plt.ylabel(r'Uber Revenue', fontsize=fs)
    plt.xlabel(r'iterations', fontsize=fs)
    plt.legend(fontsize=fs-2, loc='lower right',ncol=1)
    plt.savefig(filename+'pdf',  bbox_inches='tight', dpi=300)
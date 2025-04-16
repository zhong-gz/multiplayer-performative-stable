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
loc_cap=11
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()

BATCH=10
MAXITER=5000
np.random.seed(10)
eta=0.001 
x0=np.random.rand(2,loc_cap)

BATCH=10
MAXITER=1000
MAXITER_NE=6000
tot_rev=0
# set up the game
loc_cap=11
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()

# seed 
np.random.seed(10)
eta=0.001 
x0=np.random.rand(2,loc_cap)

## Compute Nash 
dic_sgd=ddgame.runSGD(x0,eta=0.001,BATCH=10,MAXITER=MAXITER_NE, perform_sgd=[True,True],tot_rev=0)
x_sgd=np.asarray(dic_sgd['x'])
nash=[]
for i in range(loc_cap):
    nash.append(np.mean(x_sgd[-100:,:,i],axis=0))
nash=np.asarray(nash)
    
# run all three cases
dic_sgd=ddgame.runSGD(x0,eta=0.001,BATCH=10,MAXITER=MAXITER, perform_sgd=[True,True],tot_rev=0)
dic_rgd=ddgame.runRGD(x0,eta=0.001,BATCH=10,MAXITER=MAXITER,tot_rev=0)
dic_so=ddgame.runSO(x0,eta=0.001,MAXITER=MAXITER,tot_rev=0)

x_so=np.asarray(dic_so['x'])
x_sgd=np.asarray(dic_sgd['x'])
x_rgd=np.asarray(dic_rgd['x'])

error_so=[]
error_sgd=[]
error_rgd=[]
for x,y in zip(x_so,x_sgd):
    error_so.append(la.norm(x.T-nash)**2)
    error_sgd.append(la.norm(y-nash.T)**2)
    
err_so=np.asarray(error_so)
err_sgd=np.asarray(error_sgd)

for x in x_rgd:
    error_rgd.append(la.norm(x-nash.T)**2)
    
err_rgd=np.asarray(error_rgd)

# get mean prices
x_so_avg_p1=np.mean(x_so[:,0,:],axis=1)
x_so_avg_p2=np.mean(x_so[:,1,:],axis=1)
x_sgd_avg_p1=np.mean(x_sgd[:,0,:],axis=1)
x_sgd_avg_p2=np.mean(x_sgd[:,1,:],axis=1)
x_rgd_avg_p1=np.mean(x_rgd[:,0,:],axis=1)
x_rgd_avg_p2=np.mean(x_rgd[:,1,:],axis=1)

SAVE=0
fs=24
fname='./figs_ride_share/test_end_files/prices_RGD_SGD_SO_loc.'
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ls_=['-','--']
lw=4
oss=[0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.4,0.3,0.2,0.1,0.0]
print(len(oss))

plt.plot(x_so_avg_p1[:], label='SO, p1',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(x_sgd_avg_p1[:], label='SGM (NE), p1',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:cerulean')
plt.plot(x_rgd_avg_p1[:], label='SRGM (PS), p1',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:dark blue')
plt.plot(x_so_avg_p2[:], label='SO, p2',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(x_sgd_avg_p2[:], label='SGM (NE), p2',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:cerulean')
plt.plot(x_rgd_avg_p2[:], label='SRGM (PS), p2',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:dark blue')
ax.grid(True)
ax.set_xlabel(r'iterations', fontsize=fs)
ax.legend(fontsize=fs-2,ncol=2) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
plt.tick_params(labelsize=fs-2)


plt.ylabel(r'prices', fontsize=fs)
if SAVE:
    for tag in ['png','pdf']:
        plt.savefig(fname+tag, dpi=300, transparent=True, bbox_inches='tight')

rev_so_p1=np.asarray(dic_so['revenue_total_p1'])
rev_so_p2=np.asarray(dic_so['revenue_total_p2'])
social_rev=rev_so_p1+rev_so_p2

rev_sgd_p1=np.asarray(dic_sgd['revenue_total_p1'])
rev_sgd_p2=np.asarray(dic_sgd['revenue_total_p2'])

rev_rgd_p1=np.asarray(dic_rgd['revenue_total_p1'])
rev_rgd_p2=np.asarray(dic_rgd['revenue_total_p2'])

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ls_=['-','--']
lw=4
oss=[0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.4,0.3,0.2,0.1,0.0]
print(len(oss))
mean_val=20
rev_so_p1_=running_mean(rev_so_p1,N=mean_val)
rev_so_p2_=running_mean(rev_so_p2,N=mean_val)
rev_sgd_p1_=running_mean(rev_sgd_p1,N=mean_val)
rev_sgd_p2_=running_mean(rev_sgd_p2,N=mean_val)
rev_rgd_p1_=running_mean(rev_rgd_p1,N=mean_val)
rev_rgd_p2_=running_mean(rev_rgd_p2,N=mean_val)
plt.plot(rev_so_p1_, label='SO, p1',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(rev_so_p2_, label='SO, p2',ls=ls_[1],alpha=1, lw=lw, color='xkcd:kelly green')
plt.plot(rev_sgd_p1_, label='NE, p1',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:dark blue')
plt.plot(rev_sgd_p2_, label='NE, p2',ls=ls_[1],alpha=1, lw=lw, color='xkcd:dark blue')
plt.plot(rev_rgd_p1_, label='PS, p1',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:cerulean')
plt.plot(rev_rgd_p2_, label='PS, p2',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:cerulean')
ax.grid(True)
ax.set_xlabel(r'iterations', fontsize=fs)
ax.legend(fontsize=fs-2,ncol=3) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
plt.tick_params(labelsize=fs-2)


plt.ylabel(r'revenue', fontsize=fs)
#for tag in ['png','pdf']:
#   plt.savefig(fname+tag, dpi=300, transparent=True, bbox_inches='tight')

loss_so_p1=np.asarray(dic_so['loss_p1'])
loss_so_p2=np.asarray(dic_so['loss_p2'])
social_loss=loss_so_p1+loss_so_p2

loss_sgd_p1=np.asarray(dic_sgd['loss_p1'])
loss_sgd_p2=np.asarray(dic_sgd['loss_p2'])

loss_rgd_p1=np.asarray(dic_rgd['loss_p1'])
loss_rgd_p2=np.asarray(dic_rgd['loss_p2'])

fname='./figs_ride_share/test_end_files/social_cost_rideshare.'
SAVE=0
fig, ax = plt.subplots(1, 2, figsize=(24, 7))
ls_=['-','--']
lw=4
oss=[0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.4,0.3,0.2,0.1,0.0]
print(len(oss))
mean_val=30
loss_so_p1_=running_mean(loss_so_p1,N=mean_val)
loss_so_p2_=running_mean(loss_so_p2,N=mean_val)
loss_sgd_p1_=running_mean(loss_sgd_p1,N=mean_val)
loss_sgd_p2_=running_mean(loss_sgd_p2,N=mean_val)
loss_rgd_p1_=running_mean(loss_rgd_p1,N=mean_val)
loss_rgd_p2_=running_mean(loss_rgd_p2,N=mean_val)
ax[0].plot(loss_so_p1_, label='SO, p1',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:kelly green')
ax[0].plot(loss_so_p2_, label='SO, p2',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:kelly green')
ax[0].plot(loss_so_p1_+loss_so_p2_, label='SO',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')

ax[0].plot(loss_sgd_p1_, label='NE, p1',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:dark blue')
ax[0].plot(loss_sgd_p2_, label='NE, p2',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:dark blue')
ax[0].plot(loss_sgd_p2_+loss_sgd_p1_, label='NE',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:dark blue')

ax[0].plot(loss_rgd_p1_, label='PS, p1',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:cerulean')
ax[0].plot(loss_rgd_p2_, label='PS, p2',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:cerulean')
ax[0].plot(loss_rgd_p1_+loss_rgd_p2_, label='PS',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:cerulean')

ax[0].grid(True)
ax[0].set_xlabel(r'iterations', fontsize=fs+2)

ax[0].tick_params(labelsize=fs-2)
ax[0].set_ylabel(r'loss', fontsize=fs+2)

mean_val=20
rev_so_p1_=running_mean(rev_so_p1,N=mean_val)
rev_so_p2_=running_mean(rev_so_p2,N=mean_val)
rev_sgd_p1_=running_mean(rev_sgd_p1,N=mean_val)
rev_sgd_p2_=running_mean(rev_sgd_p2,N=mean_val)
rev_rgd_p1_=running_mean(rev_rgd_p1,N=mean_val)
rev_rgd_p2_=running_mean(rev_rgd_p2,N=mean_val)
ax[1].plot(rev_so_p1_, label='SO, p1',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:kelly green')
ax[1].plot(rev_so_p2_, label='SO, p2',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:kelly green')
ax[1].plot(rev_so_p1_+rev_so_p2_, label='SO, p1+p2',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')

ax[1].plot(rev_sgd_p1_, label='NE, p1',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:dark blue')
ax[1].plot(rev_sgd_p2_, label='NE, p2',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:dark blue')
ax[1].plot(rev_sgd_p1_+rev_sgd_p2_, label='NE, p1+p2',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:dark blue')

ax[1].plot(rev_rgd_p1_, label='PS, p1',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:cerulean')
ax[1].plot(rev_rgd_p2_, label='PS, p2',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:cerulean')
ax[1].plot(rev_rgd_p1_+rev_rgd_p2_, label='PS, p1+p2',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:cerulean')
ax[1].grid(True)
ax[1].set_xlabel(r'iterations', fontsize=fs+2)
#ax[1].legend(fontsize=fs-2,ncol=3) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
ax[1].tick_params(labelsize=fs-2)
ax[1].legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.35,0.5)) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)

ax[1].set_ylabel(r'revenue', fontsize=fs+2)
plt.tight_layout()

for tag in ['png']:
    plt.savefig(fname+tag, dpi=300, transparent=True, bbox_inches='tight')

# where to store
SAVE=0
filename='./figs/test_end_files/exp_f_change_rev_demand_price10.'
fs=24
bdd=1000 # how many points to average

lyft_rev_so=running_mean(rev_so_p1,N=100) # Nash - myopic
uber_rev_so=running_mean(rev_so_p2,N=100)
lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
uber_rev_var_so = np.std(uber_rev_so[-bdd:])
var=[lyft_rev_var_so, uber_rev_var_so]

lyft_rev_sgd=running_mean(rev_sgd_p1,N=100) # Nash - myopic
uber_rev_sgd=running_mean(rev_sgd_p2,N=100)
lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
var=[lyft_rev_var_sgd, uber_rev_var_sgd]

lyft_rev_rgd=running_mean(rev_rgd_p1,N=100) # Nash - myopic
uber_rev_rgd=running_mean(rev_rgd_p2,N=100)
lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
var=[lyft_rev_var_rgd, uber_rev_var_rgd]


fig, ax = plt.subplots(1, 1, figsize=(14, 7))

data=['Lyft Revenue', 'Uber Revenue']
data_=['Lyft Demand', 'Uber Demand']
x_pos = [i for i, _ in enumerate(data)]
x_pos_ = [i for i, _ in enumerate(data_)]

tot_so=lyft_rev_so_final+uber_rev_so_final
tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final
var_so=np.std(lyft_rev_so[-bdd:]+uber_rev_so[-bdd:])
var_sgd=np.std(lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:])
var_rgd=np.std(lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:])
vals=[lyft_rev_so_final, uber_rev_so_final,tot_so,lyft_rev_sgd_final, uber_rev_sgd_final,tot_ne,lyft_rev_rgd_final,uber_rev_rgd_final,tot_ps]
var=[lyft_rev_var_so, uber_rev_var_so,var_so,lyft_rev_var_sgd, uber_rev_var_sgd,var_sgd,lyft_rev_var_rgd, uber_rev_var_rgd,var_rgd]
vals_=[lyft_rev_sgd_final, uber_rev_sgd_final]
_vals=[lyft_rev_sgd_final,uber_rev_sgd_final]

dic={key: val for key, val in zip(data,vals)}
ax.grid(True)


ax.set_ylabel("Revenue", fontsize=fs-2)
ax.set_xticks([0,1,2,3,4,5,6,7,8])
ax.set_xticklabels(['Lyft SO','Uber SO','SO','Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS'], fontsize=fs-2)
ax.bar([0,1,2,3,4,5,6,7,8], vals , yerr=var, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green','xkcd:hot pink', 
                                                       'xkcd:slate grey','xkcd:dark blue','xkcd:hot pink', 'xkcd:slate grey','xkcd:cerulean'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )

plt.tick_params(labelsize=fs-2)
ax.tick_params(labelsize=fs-2)
plt.tight_layout()

fname='./figs_ride_share/test_end_files/revenue_so_comp.'
for tag in ['pdf']:
    plt.savefig(filename+tag, dpi=300, bbox_inches='tight', transparent=True)

# where to store
filename='./figs/test_end_files/exp_f_change_rev_demand_price10.'
fs=24
bdd=1000 # how many points to average

lyft_rev_so=running_mean(rev_so_p1,N=100) # Nash - myopic
uber_rev_so=running_mean(rev_so_p2,N=100)
lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
uber_rev_var_so = np.std(uber_rev_so[-bdd:])
var=[lyft_rev_var_so, uber_rev_var_so]

lyft_rev_sgd=running_mean(rev_sgd_p1,N=100) # Nash - myopic
uber_rev_sgd=running_mean(rev_sgd_p2,N=100)
lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
var=[lyft_rev_var_sgd, uber_rev_var_sgd]

lyft_rev_rgd=running_mean(rev_rgd_p1,N=100) # Nash - myopic
uber_rev_rgd=running_mean(rev_rgd_p2,N=100)
lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
var=[lyft_rev_var_rgd, uber_rev_var_rgd]

fig, ax = plt.subplots(1, 3, figsize=(10, 7), sharey=True)

data=['Lyft Revenue', 'Uber Revenue']
data_=['Lyft Demand', 'Uber Demand']
x_pos = [i for i, _ in enumerate(data)]
x_pos_ = [i for i, _ in enumerate(data_)]

tot_so=lyft_rev_so_final+uber_rev_so_final
tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final
var_so=np.std(lyft_rev_so[-bdd:]+uber_rev_so[-bdd:])
var_sgd=np.std(lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:])
var_rgd=np.std(lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:])

vals_so=[lyft_rev_so_final, uber_rev_so_final,tot_so]
vals_ne=[lyft_rev_sgd_final, uber_rev_sgd_final,tot_ne]
vals_ps=[lyft_rev_rgd_final,uber_rev_rgd_final,tot_ps]

var_so=[lyft_rev_var_so, uber_rev_var_so,var_so]
var_ne=[lyft_rev_var_sgd, uber_rev_var_sgd,var_sgd]
var_ps=[lyft_rev_var_rgd, uber_rev_var_rgd,var_rgd]


dic={key: val for key, val in zip(data,vals)}
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_ylabel("Revenue", fontsize=fs-2)


ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax[0].bar([0,1,2], vals_so , yerr=var_so, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )

ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax[1].bar([0,1,2], vals_ne , yerr=var_ne, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:dark blue'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )

ax[2].set_xticks([0,1,2])
ax[2].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax[2].bar([0,1,2], vals_ps , yerr=var_ps, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:cerulean'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )

ax[2].set_title("PS", fontsize=fs-2)
ax[1].set_title("NE", fontsize=fs-2)
ax[0].set_title("SO", fontsize=fs-2)
plt.tick_params(labelsize=fs-2)
ax[0].tick_params(labelsize=fs-2)
ax[2].tick_params(labelsize=fs-2)
ax[1].tick_params(labelsize=fs-2)
plt.tight_layout()
SAVE=0
filename='./figs/test_end_files/revenue_so_comp_alt.'
if SAVE:
    for tag in ['pdf', 'png']:
        plt.savefig(filename+tag, dpi=300, bbox_inches='tight', transparent=True)

# where to store
filename='./figs/test_end_files/exp_f_change_rev_demand_price10.'
fs=24
bdd=1000 # how many points to average

lyft_rev_so=running_mean(loss_so_p1,N=100) # Nash - myopic
uber_rev_so=running_mean(loss_so_p2,N=100)
lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
uber_rev_var_so = np.std(uber_rev_so[-bdd:])
var=[lyft_rev_var_so, uber_rev_var_so]

lyft_rev_sgd=running_mean(loss_sgd_p1,N=100) # Nash - myopic
uber_rev_sgd=running_mean(loss_sgd_p2,N=100)
lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
var=[lyft_rev_var_sgd, uber_rev_var_sgd]


lyft_rev_rgd=running_mean(loss_rgd_p1,N=100) # Nash - myopic
uber_rev_rgd=running_mean(loss_rgd_p2,N=100)
lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
var=[lyft_rev_var_rgd, uber_rev_var_rgd]



fig, ax = plt.subplots(1, 3, figsize=(10, 7), sharey=True)

data=['Lyft Revenue', 'Uber Revenue']
data_=['Lyft Demand', 'Uber Demand']
x_pos = [i for i, _ in enumerate(data)]
x_pos_ = [i for i, _ in enumerate(data_)]

tot_so=lyft_rev_so_final+uber_rev_so_final
tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final

poa_ps=tot_ps/tot_so
poa_ne=tot_ne/tot_so

loss_tot_vec=lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:]
poa_vec_ne=[l/tot_so for l in loss_tot_vec]
poa_ne_mean=np.mean(np.asarray(poa_vec_ne))
poa_ne_var=np.std(np.asarray(poa_vec_ne))

loss_tot_vec=lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:]
poa_vec_ps=[l/tot_so for l in loss_tot_vec]
poa_ps_mean=np.mean(np.asarray(poa_vec_ps))
poa_ps_var=np.std(np.asarray(poa_vec_ps))

var_so=np.std(lyft_rev_so[-bdd:]+uber_rev_so[-bdd:])
var_sgd=np.std(lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:])
var_rgd=np.std(lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:])

vals_so=[lyft_rev_so_final, uber_rev_so_final,tot_so]
vals_ne=[lyft_rev_sgd_final, uber_rev_sgd_final,tot_ne]
vals_ps=[lyft_rev_rgd_final,uber_rev_rgd_final,tot_ps]

var_so=[lyft_rev_var_so, uber_rev_var_so,var_so]
var_ne=[lyft_rev_var_sgd, uber_rev_var_sgd,var_sgd]
var_ps=[lyft_rev_var_rgd, uber_rev_var_rgd,var_rgd]


dic={key: val for key, val in zip(data,vals)}
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

ax[0].set_ylabel("Loss", fontsize=fs-2)


ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax[0].bar([0,1,2], vals_so , yerr=var_so, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )

ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax[1].bar([0,1,2], vals_ne , yerr=var_ne, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:dark blue'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )

ax[2].set_xticks([0,1,2])
ax[2].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax[2].bar([0,1,2], vals_ps , yerr=var_ps, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:cerulean'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )
#ax[2].set_title("Performatively Stable", fontsize=fs-2)
#ax[1].set_title("Nash Equilibrium", fontsize=fs-2)
#ax[0].set_title("Social Optimum", fontsize=fs-2)

ax[2].set_title("PS", fontsize=fs-2)
ax[1].set_title("NE", fontsize=fs-2)
ax[0].set_title("SO", fontsize=fs-2)

plt.tick_params(labelsize=fs-2)
ax[0].tick_params(labelsize=fs-2)
ax[2].tick_params(labelsize=fs-2)
ax[1].tick_params(labelsize=fs-2)
plt.tight_layout()
SAVE=0
filename='./figs/test_end_files/loss_so_comp_alt.'
if SAVE:
    for tag in ['pdf', 'png']:
        plt.savefig(filename+tag, dpi=300, bbox_inches='tight', transparent=True)

# where to store
filename='./figs/test_end_files/exp_f_change_rev_demand_price10.'
fs=24
bdd=1000 # how many points to average

lyft_rev_so=running_mean(loss_so_p1,N=100) # Nash - myopic
uber_rev_so=running_mean(loss_so_p2,N=100)
lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
uber_rev_var_so = np.std(uber_rev_so[-bdd:])
var=[lyft_rev_var_so, uber_rev_var_so]

lyft_rev_sgd=running_mean(loss_sgd_p1,N=100) # Nash - myopic
uber_rev_sgd=running_mean(loss_sgd_p2,N=100)
lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
var=[lyft_rev_var_sgd, uber_rev_var_sgd]


lyft_rev_rgd=running_mean(loss_rgd_p1,N=100) # Nash - myopic
uber_rev_rgd=running_mean(loss_rgd_p2,N=100)
lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
var=[lyft_rev_var_rgd, uber_rev_var_rgd]



fig, ax = plt.subplots(1, 1, figsize=(5, 7), sharey=True)

data=['Lyft Revenue', 'Uber Revenue']
data_=['Lyft Demand', 'Uber Demand']
x_pos = [i for i, _ in enumerate(data)]
x_pos_ = [i for i, _ in enumerate(data_)]

tot_so=lyft_rev_so_final+uber_rev_so_final
tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final

poa_ps=tot_ps/tot_so
poa_ne=tot_ne/tot_so

loss_tot_vec=lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:]
poa_vec_ne=[l/tot_so for l in loss_tot_vec]
poa_ne_mean=np.mean(np.asarray(poa_vec_ne))
poa_ne_var=np.std(np.asarray(poa_vec_ne))

loss_tot_vec=lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:]
poa_vec_ps=[l/tot_so for l in loss_tot_vec]
poa_ps_mean=np.mean(np.asarray(poa_vec_ps))
poa_ps_var=np.std(np.asarray(poa_vec_ps))

poa=[poa_ps_mean, poa_ne_mean]
poa_var=[poa_ps_var, poa_ne_var]


dic={key: val for key, val in zip(data,vals)}
ax.grid(True)


ax.set_ylabel("price of anarchy (PoA)", fontsize=fs)


ax.set_xticks([0,1])
ax.set_xticklabels(['PS','NE'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
ax.bar([0,1], poa , yerr=poa_var, color=['xkcd:cerulean', 'xkcd:dark blue'],
          error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
        )



plt.tick_params(labelsize=fs-2)

plt.tight_layout()
SAVE=0
filename='./figs/test_end_files/poa_rideshare.'
if SAVE:
    for tag in ['pdf', 'png']:
        plt.savefig(filename+tag, dpi=300, bbox_inches='tight', transparent=True)
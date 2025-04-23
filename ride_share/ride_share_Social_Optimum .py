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
# from utilsrm_modified import *
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
# seed 
np.random.seed(10)
figuresize=(13, 7)
loc_cap=11
eta=0.001 
x0=np.random.rand(2,loc_cap)
gamma = 2.1
price_index = 0
BATCH=10
MAXITER=1000
MAXITER_NE=6000
tot_rev=1
# set up the game
loc_lst_index=list(range(0,loc_cap))
price_lst_index=list(range(0,3))
ddgame=ddrideshare(loc_lst_index,price_lst_index,seed=2,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10)
ddgame.setup_distribution()

# ## Compute Nash 
A1_hat = np.diag(-10*np.random.rand(np.shape(ddgame.A1)[1]))
Ac1_hat = np.diag(2*np.random.rand(np.shape(ddgame.Ac1)[1]))
A2_hat = np.diag(-10*np.random.rand(np.shape(ddgame.A2)[1]))
Ac2_hat = np.diag(2*np.random.rand(np.shape(ddgame.Ac2)[1]))
A_dic={}
A_dic['A1_hat']=A1_hat
A_dic['Ac1_hat']=Ac1_hat
A_dic['A2_hat']=A2_hat
A_dic['Ac2_hat']=Ac2_hat

# run all cases
# dic_sgd=ddgame.runSGD(x0,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER, perform_sgd=[True,True],tot_rev=tot_rev)
# dic_so=ddgame.runSO(x0,eta=eta,price_index=price_index,MAXITER=MAXITER,tot_rev=tot_rev)
dic_agd=ddgame.runAGD(x0,A_dic,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev)
dic_rgd=ddgame.runRGD(x0,eta=eta,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER,tot_rev=tot_rev)
dic_rr=ddgame.runRR(gamma = gamma,price_index=price_index,BATCH=BATCH,MAXITER=MAXITER, perform_rr=[True,True],tot_rev=tot_rev)

x_agd=np.asarray(dic_agd['x'])
x_rgd=np.asarray(dic_rgd['x'])
x_rr=np.asarray(dic_rr['x'])


# get mean prices
price_mean = price_index*5+10
x_agd_avg_p1=np.mean(x_agd[:,0,:],axis=1)+price_mean
x_agd_avg_p2=np.mean(x_agd[:,1,:],axis=1)+price_mean
x_rgd_avg_p1=np.mean(x_rgd[:,0,:],axis=1)+price_mean
x_rgd_avg_p2=np.mean(x_rgd[:,1,:],axis=1)+price_mean
x_rr_avg_p1=np.mean(x_rr[:,0,:],axis=1)+price_mean
x_rr_avg_p2=np.mean(x_rr[:,1,:],axis=1)+price_mean

fs=24
fname='./figs_ride_share/test_end_files/prices_RGD_SGD_SO_loc.'
# fig, ax = plt.subplots(1, 1, figsize=figuresize)
plt.figure(figsize=figuresize)
ls_=['-','--']
lw=4

plt.plot(x_agd_avg_p1[:], label='AGM, Lyft',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(x_agd_avg_p2[:], label='AGM, Uber',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(x_rgd_avg_p1[:], label='RGD, Lyft',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:dark blue')
plt.plot(x_rgd_avg_p2[:], label='RGD, Uber',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:dark blue')
plt.plot(x_rr_avg_p1[:], label='RR, Lyft',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:light orange')
plt.plot(x_rr_avg_p2[:], label='RR, Uber',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:light orange')
plt.grid(True)
plt.xlabel(r'iterations', fontsize=fs)
# plt.legend(fontsize=fs-2,ncol=2) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.35,0.5))
plt.tick_params(labelsize=fs-2)
plt.ylabel(r'prices', fontsize=fs)
plt.tight_layout()
plt.savefig(fname+'pdf', dpi=300, transparent=True, bbox_inches='tight')

rev_agd_p1=np.asarray(dic_agd['revenue_total_p1'])
rev_agd_p2=np.asarray(dic_agd['revenue_total_p2'])

rev_rgd_p1=np.asarray(dic_rgd['revenue_total_p1'])
rev_rgd_p2=np.asarray(dic_rgd['revenue_total_p2'])

rev_rr_p1=np.asarray(dic_rr['revenue_total_p1'])
rev_rr_p2=np.asarray(dic_rr['revenue_total_p2'])

fname='./figs_ride_share/test_end_files/revenue_RGD_SGD_SO_loc.'
# fig, ax = plt.subplots(1, 1, figsize=figuresize)
plt.figure(figsize=figuresize)
ls_=['-','--']
lw=4
mean_val=20
rev_agd_p1_=running_mean(rev_agd_p1,N=mean_val)
rev_agd_p2_=running_mean(rev_agd_p2,N=mean_val)
rev_rgd_p1_=running_mean(rev_rgd_p1,N=mean_val)
rev_rgd_p2_=running_mean(rev_rgd_p2,N=mean_val)
rev_rr_p1_=running_mean(rev_rr_p1,N=mean_val)
rev_rr_p2_=running_mean(rev_rr_p2,N=mean_val)
plt.plot(rev_agd_p1_, label='AGM, Lyft',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(rev_agd_p2_, label='AGM, Uber',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:kelly green')
plt.plot(rev_rgd_p1_, label='RGD, Lyft',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:cerulean')
plt.plot(rev_rgd_p2_, label='RGD, Uber',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:cerulean')
plt.plot(rev_rr_p1_, label= 'RR,  Lyft',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:light orange')
plt.plot(rev_rr_p2_, label= 'RR,  Uber',ls=ls_[1],alpha=1.0, lw=lw, color='xkcd:light orange')
plt.grid(True)
plt.xlabel(r'iterations', fontsize=fs)
# plt.legend(fontsize=fs-2,ncol=3) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.35,0.5))
plt.tick_params(labelsize=fs-2)
plt.ylabel(r'revenue', fontsize=fs)
plt.tight_layout()
plt.savefig(fname+'pdf', dpi=300, transparent=True, bbox_inches='tight')

loss_agd_p1=np.asarray(dic_agd['loss_p1'])
loss_agd_p2=np.asarray(dic_agd['loss_p2'])

loss_rgd_p1=np.asarray(dic_rgd['loss_p1'])
loss_rgd_p2=np.asarray(dic_rgd['loss_p2'])

loss_rr_p1=np.asarray(dic_rr['loss_p1'])
loss_rr_p2=np.asarray(dic_rr['loss_p2'])

fname='./figs_ride_share/test_end_files/social_cost_rideshare.'
# fig, ax = plt.subplots(1, 2, figsize=(24, 7))
# fig, ax = plt.subplots(1, 1, figsize=(24, 7))
# new_figuresize = (figuresize[0] + 5,) + figuresize[1:]
plt.figure(figsize=figuresize)
ls_=['-','--']
lw=4
mean_val=30
loss_agd_p1_=running_mean(loss_agd_p1,N=mean_val)
loss_agd_p2_=running_mean(loss_agd_p2,N=mean_val)
loss_rgd_p1_=running_mean(loss_rgd_p1,N=mean_val)
loss_rgd_p2_=running_mean(loss_rgd_p2,N=mean_val)
loss_rr_p1_=running_mean(loss_rr_p1,N=mean_val)
loss_rr_p2_=running_mean(loss_rr_p2,N=mean_val)

plt.plot(rev_agd_p1_, label='AGM, Lyft',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:kelly green')
plt.plot(rev_agd_p2_, label='AGM, Uber',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:kelly green')
plt.plot(rev_agd_p1_+rev_agd_p2_, label='AGM',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:kelly green')

plt.plot(rev_rgd_p1_, label='RGD, Lyft',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:cerulean')
plt.plot(rev_rgd_p2_, label='RGD, Uber',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:cerulean')
plt.plot(rev_rgd_p1_+rev_rgd_p2_, label='RGD, Lyft+Uber',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:cerulean')

plt.plot(rev_rr_p1_, label='RR, Lyft',ls=ls_[0],alpha=0.5, lw=lw, color='xkcd:light orange')
plt.plot(rev_rr_p2_, label='RR, Uber',ls=ls_[1],alpha=0.5, lw=lw, color='xkcd:light orange')
plt.plot(rev_rr_p1_+rev_rr_p2_, label='RR, Lyft+Uber',ls=ls_[0],alpha=1.0, lw=lw, color='xkcd:light orange')
plt.grid(True)
plt.xlabel(r'iterations', fontsize=fs+2)
#plt.legend(fontsize=fs-2,ncol=3) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
plt.tick_params(labelsize=fs-2)
plt.legend(fontsize=fs-2,ncol=1, loc='right',bbox_to_anchor=(1.55,0.5)) #loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)

plt.ylabel(r'revenue', fontsize=fs+2)
plt.tight_layout()
plt.savefig(fname+'pdf', dpi=300, transparent=True, bbox_inches='tight')

# # where to store
# filename='./figs_ride_share/test_end_files/exp_f_change_rev_demand_price10.'
# fs=24
# bdd=1000 # how many points to average

# lyft_rev_so=running_mean(rev_so_p1,N=100) # Nash - myopic
# uber_rev_so=running_mean(rev_so_p2,N=100)
# lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
# uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
# lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
# uber_rev_var_so = np.std(uber_rev_so[-bdd:])
# var=[lyft_rev_var_so, uber_rev_var_so]

# lyft_rev_sgd=running_mean(rev_sgd_p1,N=100) # Nash - myopic
# uber_rev_sgd=running_mean(rev_sgd_p2,N=100)
# lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
# uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
# lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
# uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
# var=[lyft_rev_var_sgd, uber_rev_var_sgd]

# lyft_rev_rgd=running_mean(rev_rgd_p1,N=100) # Nash - myopic
# uber_rev_rgd=running_mean(rev_rgd_p2,N=100)
# lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
# uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
# lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
# uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
# var=[lyft_rev_var_rgd, uber_rev_var_rgd]

# lyfy_rev_rr=running_mean(rev_rr_p1,N=100) # Nash - myopic
# uber_rev_rr=running_mean(rev_rr_p2,N=100)
# lyft_rev_rr_final=np.mean(lyfy_rev_rr[-bdd:])
# uber_rev_rr_final=np.mean(uber_rev_rr[-bdd:])
# lyft_rev_var_rr = np.std(lyfy_rev_rr[-bdd:])
# uber_rev_var_rr = np.std(uber_rev_rr[-bdd:])
# var=[lyft_rev_var_rr, uber_rev_var_rr]

# fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# data=['Lyft Revenue', 'Uber Revenue']
# data_=['Lyft Demand', 'Uber Demand']
# x_pos = [i for i, _ in enumerate(data)]
# x_pos_ = [i for i, _ in enumerate(data_)]

# tot_so=lyft_rev_so_final+uber_rev_so_final
# tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
# tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final
# tot_rr=lyft_rev_rr_final+uber_rev_rr_final
# var_so=np.std(lyft_rev_so[-bdd:]+uber_rev_so[-bdd:])
# var_sgd=np.std(lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:])
# var_rgd=np.std(lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:])
# var_rr=np.std(lyfy_rev_rr[-bdd:]+uber_rev_rr[-bdd:])
# vals=[lyft_rev_so_final, uber_rev_so_final,tot_so,lyft_rev_sgd_final, uber_rev_sgd_final,tot_ne,lyft_rev_rgd_final,uber_rev_rgd_final,tot_ps,lyft_rev_rr_final,uber_rev_rr_final,tot_rr]
# var =[lyft_rev_var_so, uber_rev_var_so,var_so,lyft_rev_var_sgd, uber_rev_var_sgd,var_sgd,lyft_rev_var_rgd, uber_rev_var_rgd,var_rgd,lyft_rev_var_rr, uber_rev_var_rr,var_rr]
# vals_=[lyft_rev_sgd_final, uber_rev_sgd_final]
# _vals=[lyft_rev_sgd_final,uber_rev_sgd_final]

# dic={key: val for key, val in zip(data,vals)}
# ax.grid(True)

# ax.set_ylabel("Revenue", fontsize=fs-2)
# ax.set_xticks([0,1,2,3,4,5,6,7,8])
# ax.set_xticklabels(['Lyft SO','Uber SO','SO','Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS','Lyft RR','Uber RR', 'RR'], fontsize=fs-2)
# ax.bar([0,1,2,3,4,5,6,7,8,9,10,11], vals , yerr=var, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green','xkcd:hot pink', 
#                                                        'xkcd:slate grey','xkcd:dark blue','xkcd:hot pink', 'xkcd:slate grey','xkcd:cerulean',
#                                                        'xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )

# plt.tick_params(labelsize=fs-2)
# ax.tick_params(labelsize=fs-2)
# plt.tight_layout()

# fname='./figs_ride_share/test_end_files/revenue_so_comp.'
# plt.savefig(filename+'pdf', dpi=300, bbox_inches='tight', transparent=True)

# # where to store
# filename='./figs_ride_share/test_end_files/exp_f_change_rev_demand_price10.'
# fs=24
# bdd=1000 # how many points to average

# lyft_rev_so=running_mean(rev_so_p1,N=100) # Nash - myopic
# uber_rev_so=running_mean(rev_so_p2,N=100)
# lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
# uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
# lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
# uber_rev_var_so = np.std(uber_rev_so[-bdd:])
# var=[lyft_rev_var_so, uber_rev_var_so]

# lyft_rev_sgd=running_mean(rev_sgd_p1,N=100) # Nash - myopic
# uber_rev_sgd=running_mean(rev_sgd_p2,N=100)
# lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
# uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
# lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
# uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
# var=[lyft_rev_var_sgd, uber_rev_var_sgd]

# lyft_rev_rgd=running_mean(rev_rgd_p1,N=100) # Nash - myopic
# uber_rev_rgd=running_mean(rev_rgd_p2,N=100)
# lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
# uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
# lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
# uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
# var=[lyft_rev_var_rgd, uber_rev_var_rgd]

# fig, ax = plt.subplots(1, 3, figsize=figuresize, sharey=True)

# data=['Lyft Revenue', 'Uber Revenue']
# data_=['Lyft Demand', 'Uber Demand']
# x_pos = [i for i, _ in enumerate(data)]
# x_pos_ = [i for i, _ in enumerate(data_)]

# tot_so=lyft_rev_so_final+uber_rev_so_final
# tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
# tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final
# var_so=np.std(lyft_rev_so[-bdd:]+uber_rev_so[-bdd:])
# var_sgd=np.std(lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:])
# var_rgd=np.std(lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:])

# vals_so=[lyft_rev_so_final, uber_rev_so_final,tot_so]
# vals_ne=[lyft_rev_sgd_final, uber_rev_sgd_final,tot_ne]
# vals_ps=[lyft_rev_rgd_final,uber_rev_rgd_final,tot_ps]

# var_so=[lyft_rev_var_so, uber_rev_var_so,var_so]
# var_ne=[lyft_rev_var_sgd, uber_rev_var_sgd,var_sgd]
# var_ps=[lyft_rev_var_rgd, uber_rev_var_rgd,var_rgd]


# dic={key: val for key, val in zip(data,vals)}
# ax[0].grid(True)
# ax[1].grid(True)
# ax[2].grid(True)
# ax[0].set_ylabel("Revenue", fontsize=fs-2)


# ax[0].set_xticks([0,1,2])
# ax[0].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax[0].bar([0,1,2], vals_so , yerr=var_so, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )

# ax[1].set_xticks([0,1,2])
# ax[1].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax[1].bar([0,1,2], vals_ne , yerr=var_ne, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:dark blue'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )

# ax[2].set_xticks([0,1,2])
# ax[2].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax[2].bar([0,1,2], vals_ps , yerr=var_ps, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:cerulean'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )

# ax[2].set_title("PS", fontsize=fs-2)
# ax[1].set_title("NE", fontsize=fs-2)
# ax[0].set_title("SO", fontsize=fs-2)
# plt.tick_params(labelsize=fs-2)
# ax[0].tick_params(labelsize=fs-2)
# ax[2].tick_params(labelsize=fs-2)
# ax[1].tick_params(labelsize=fs-2)
# plt.tight_layout()
# filename='./figs_ride_share/test_end_files/revenue_so_comp_alt.'
# plt.savefig(filename+'pdf', dpi=300, bbox_inches='tight', transparent=True)

# # where to store
# filename='./figs_ride_share/test_end_files/exp_f_change_rev_demand_price10.'
# fs=24
# bdd=1000 # how many points to average

# lyft_rev_so=running_mean(loss_so_p1,N=100) # Nash - myopic
# uber_rev_so=running_mean(loss_so_p2,N=100)
# lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
# uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
# lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
# uber_rev_var_so = np.std(uber_rev_so[-bdd:])
# var=[lyft_rev_var_so, uber_rev_var_so]

# lyft_rev_sgd=running_mean(loss_sgd_p1,N=100) # Nash - myopic
# uber_rev_sgd=running_mean(loss_sgd_p2,N=100)
# lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
# uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
# lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
# uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
# var=[lyft_rev_var_sgd, uber_rev_var_sgd]


# lyft_rev_rgd=running_mean(loss_rgd_p1,N=100) # Nash - myopic
# uber_rev_rgd=running_mean(loss_rgd_p2,N=100)
# lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
# uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
# lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
# uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
# var=[lyft_rev_var_rgd, uber_rev_var_rgd]



# fig, ax = plt.subplots(1, 3, figsize=figuresize, sharey=True)

# data=['Lyft Revenue', 'Uber Revenue']
# data_=['Lyft Demand', 'Uber Demand']
# x_pos = [i for i, _ in enumerate(data)]
# x_pos_ = [i for i, _ in enumerate(data_)]

# tot_so=lyft_rev_so_final+uber_rev_so_final
# tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
# tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final

# poa_ps=tot_ps/tot_so
# poa_ne=tot_ne/tot_so

# loss_tot_vec=lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:]
# poa_vec_ne=[l/tot_so for l in loss_tot_vec]
# poa_ne_mean=np.mean(np.asarray(poa_vec_ne))
# poa_ne_var=np.std(np.asarray(poa_vec_ne))

# loss_tot_vec=lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:]
# poa_vec_ps=[l/tot_so for l in loss_tot_vec]
# poa_ps_mean=np.mean(np.asarray(poa_vec_ps))
# poa_ps_var=np.std(np.asarray(poa_vec_ps))

# var_so=np.std(lyft_rev_so[-bdd:]+uber_rev_so[-bdd:])
# var_sgd=np.std(lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:])
# var_rgd=np.std(lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:])

# vals_so=[lyft_rev_so_final, uber_rev_so_final,tot_so]
# vals_ne=[lyft_rev_sgd_final, uber_rev_sgd_final,tot_ne]
# vals_ps=[lyft_rev_rgd_final,uber_rev_rgd_final,tot_ps]

# var_so=[lyft_rev_var_so, uber_rev_var_so,var_so]
# var_ne=[lyft_rev_var_sgd, uber_rev_var_sgd,var_sgd]
# var_ps=[lyft_rev_var_rgd, uber_rev_var_rgd,var_rgd]


# dic={key: val for key, val in zip(data,vals)}
# ax[0].grid(True)
# ax[1].grid(True)
# ax[2].grid(True)

# ax[0].set_ylabel("Loss", fontsize=fs-2)


# ax[0].set_xticks([0,1,2])
# ax[0].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax[0].bar([0,1,2], vals_so , yerr=var_so, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:kelly green'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )

# ax[1].set_xticks([0,1,2])
# ax[1].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax[1].bar([0,1,2], vals_ne , yerr=var_ne, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:dark blue'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )

# ax[2].set_xticks([0,1,2])
# ax[2].set_xticklabels(['Lyft','Uber','Total'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax[2].bar([0,1,2], vals_ps , yerr=var_ps, color=['xkcd:hot pink', 'xkcd:slate grey','xkcd:cerulean'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )
# #ax[2].set_title("Performatively Stable", fontsize=fs-2)
# #ax[1].set_title("Nash Equilibrium", fontsize=fs-2)
# #ax[0].set_title("Social Optimum", fontsize=fs-2)

# ax[2].set_title("PS", fontsize=fs-2)
# ax[1].set_title("NE", fontsize=fs-2)
# ax[0].set_title("SO", fontsize=fs-2)

# plt.tick_params(labelsize=fs-2)
# ax[0].tick_params(labelsize=fs-2)
# ax[2].tick_params(labelsize=fs-2)
# ax[1].tick_params(labelsize=fs-2)
# plt.tight_layout()
# filename='./figs_ride_share/test_end_files/loss_so_comp_alt.'
# plt.savefig(filename+'pdf', dpi=300, bbox_inches='tight', transparent=True)

# # where to store
# filename='./figs_ride_share/test_end_files/exp_f_change_rev_demand_price10.'
# fs=24
# bdd=1000 # how many points to average

# lyft_rev_so=running_mean(loss_so_p1,N=100) # Nash - myopic
# uber_rev_so=running_mean(loss_so_p2,N=100)
# lyft_rev_so_final=np.mean(lyft_rev_so[-bdd:])
# uber_rev_so_final=np.mean(uber_rev_so[-bdd:])
# lyft_rev_var_so = np.std(lyft_rev_so[-bdd:])
# uber_rev_var_so = np.std(uber_rev_so[-bdd:])
# var=[lyft_rev_var_so, uber_rev_var_so]

# lyft_rev_sgd=running_mean(loss_sgd_p1,N=100) # Nash - myopic
# uber_rev_sgd=running_mean(loss_sgd_p2,N=100)
# lyft_rev_sgd_final=np.mean(lyft_rev_sgd[-bdd:])
# uber_rev_sgd_final=np.mean(uber_rev_sgd[-bdd:])
# lyft_rev_var_sgd = np.std(lyft_rev_sgd[-bdd:])
# uber_rev_var_sgd = np.std(uber_rev_sgd[-bdd:])
# var=[lyft_rev_var_sgd, uber_rev_var_sgd]


# lyft_rev_rgd=running_mean(loss_rgd_p1,N=100) # Nash - myopic
# uber_rev_rgd=running_mean(loss_rgd_p2,N=100)
# lyft_rev_rgd_final=np.mean(lyft_rev_rgd[-bdd:])
# uber_rev_rgd_final=np.mean(uber_rev_rgd[-bdd:])
# lyft_rev_var_rgd = np.std(lyft_rev_rgd[-bdd:])
# uber_rev_var_rgd = np.std(uber_rev_rgd[-bdd:])
# var=[lyft_rev_var_rgd, uber_rev_var_rgd]



# fig, ax = plt.subplots(1, 1, figsize=(5, 7), sharey=True)

# data=['Lyft Revenue', 'Uber Revenue']
# data_=['Lyft Demand', 'Uber Demand']
# x_pos = [i for i, _ in enumerate(data)]
# x_pos_ = [i for i, _ in enumerate(data_)]

# tot_so=lyft_rev_so_final+uber_rev_so_final
# tot_ne=lyft_rev_sgd_final+uber_rev_sgd_final
# tot_ps=lyft_rev_rgd_final+uber_rev_rgd_final

# poa_ps=tot_ps/tot_so
# poa_ne=tot_ne/tot_so

# loss_tot_vec=lyft_rev_sgd[-bdd:]+uber_rev_sgd[-bdd:]
# poa_vec_ne=[l/tot_so for l in loss_tot_vec]
# poa_ne_mean=np.mean(np.asarray(poa_vec_ne))
# poa_ne_var=np.std(np.asarray(poa_vec_ne))

# loss_tot_vec=lyft_rev_rgd[-bdd:]+uber_rev_rgd[-bdd:]
# poa_vec_ps=[l/tot_so for l in loss_tot_vec]
# poa_ps_mean=np.mean(np.asarray(poa_vec_ps))
# poa_ps_var=np.std(np.asarray(poa_vec_ps))

# poa=[poa_ps_mean, poa_ne_mean]
# poa_var=[poa_ps_var, poa_ne_var]


# dic={key: val for key, val in zip(data,vals)}
# ax.grid(True)


# ax.set_ylabel("price of anarchy (PoA)", fontsize=fs)


# ax.set_xticks([0,1])
# ax.set_xticklabels(['PS','NE'], fontsize=fs-2) #'Lyft NE','Uber NE','NE','Lyft PS','Uber PS','PS']
# ax.bar([0,1], poa , yerr=poa_var, color=['xkcd:cerulean', 'xkcd:dark blue'],
#           error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'),
#         )



# plt.tick_params(labelsize=fs-2)

# plt.tight_layout()
# filename='./figs_ride_share/test_end_files/poa_rideshare.'
# plt.savefig(filename+'pdf', dpi=300, bbox_inches='tight', transparent=True)
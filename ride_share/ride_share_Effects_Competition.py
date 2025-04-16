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

## Experiment 3: Effects of Competition
# run base nash case
dic_sgd=ddgame.runSGD(x0,eta=0.001,BATCH=BATCH,MAXITER=MAXITER, perform_sgd=[True,True], MYOPIC=False, tot_rev=0)
setting='Lyft and Uber Myopic'

if setting=='Lyft and Uber Myopic':
    MYOPIC=True
    perform_sgd=[False,False]
elif setting=='Uber Myopic Only':
    MYOPIC=True
    perform_sgd=[True,False]
elif setting=='Lyft Myopic Only':
    MYOPIC=True
    perform_sgd=[False,True]
elif setting=='Lyft and Uber Partially Myopic':
    MYOPIC=False
    perform_sgd=[False,False]
elif setting=='Uber Partially Myopic Only':
    MYOPIC=False
    perform_sgd=[True,False]
elif setting=='Lyft Partially Myopic Only':
    MYOPIC=False
    perform_sgd=[False,True]

dic_sgd_ignore=ddgame.runSGD(x0,eta=0.001,BATCH=BATCH,MAXITER=MAXITER, perform_sgd=perform_sgd, MYOPIC=MYOPIC, tot_rev=0) 

# store relevant data and compute data frames
rev_ig_p1_loc=dic_sgd_ignore['revenue_by_loc_p1']
rev_ig_p2_loc=dic_sgd_ignore['revenue_by_loc_p2']

rev_p1_loc=dic_sgd['revenue_by_loc_p1']
rev_p2_loc=dic_sgd['revenue_by_loc_p2']

rev_ig_p1=dic_sgd_ignore['revenue_total_p1']
rev_ig_p2=dic_sgd_ignore['revenue_total_p2']

rev_p1=dic_sgd['revenue_total_p1']
rev_p2=dic_sgd['revenue_total_p2']

demand_ig_p1=dic_sgd_ignore['demand_p1']
demand_ig_p2=dic_sgd_ignore['demand_p2']

demand_p1=dic_sgd['demand_p1']
demand_p2=dic_sgd['demand_p2']

x_sgd=np.asarray(dic_sgd['x'])
x_sgd_ig=np.asarray(dic_sgd_ignore['x'])

lyft_avg_price=np.mean(x_sgd[-100:,0,:], axis=0)
uber_avg_price=np.mean(x_sgd[-100:,1,:], axis=0)

df_all_u,df_all_l=ddgame.get_dataframe_for_plot(rev_ig_p1, rev_ig_p2, demand_ig_p1, demand_ig_p2, rev_ig_p1_loc, rev_ig_p2_loc, 
                                                rev_p1, rev_p2, demand_p1, demand_p2, rev_p1_loc, rev_p2_loc,x_sgd,x_sgd_ig,
                                                shift=4900, shift_amt=0.002, mean_back=100, scale=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
lyft_avg_price=np.mean(x_sgd[:,0,:], axis=1)
uber_avg_price=np.mean(x_sgd[:,1,:], axis=1)

lyft_avg_price_=np.mean(x_sgd_ig[:,0,:], axis=1)
uber_avg_price_=np.mean(x_sgd_ig[:,1,:], axis=1)

op=[1.0,0.5]
plt.plot(lyft_avg_price, color='xkcd:hot pink', linewidth=4, label='Lyft')
plt.plot(uber_avg_price, color='xkcd:slate grey', linewidth=4, label='Uber')
plt.plot(lyft_avg_price_, color='xkcd:hot pink', linewidth=4, label='Lyft', alpha=0.5)
plt.plot(uber_avg_price_, color='xkcd:slate grey', linewidth=4, label='Uber', alpha=0.5)
ax.grid(True)
ax.legend(fontsize=20,loc='center',bbox_to_anchor=(0.5,-0.2),ncol=6)
plt.tick_params(labelsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('average price', fontsize=20)

# where to store
filename='./figs/test_end_files/exp_f_change_rev_demand_price10.'
SAVE=1
fs=24
bdd=100 # how many points to average

lyft_rev=running_mean(rev_p1-rev_ig_p1,N=100) # Nash - myopic
uber_rev=running_mean(rev_p2-rev_ig_p2,N=100)
lyft_rev_final=np.mean(lyft_rev[-bdd:])
uber_rev_final=np.mean(uber_rev[-bdd:])
lyft_rev_var = np.std(lyft_rev[-bdd:])
uber_rev_var = np.std(uber_rev[-bdd:])
var=[lyft_rev_var, uber_rev_var]

lyft_demand=running_mean(demand_p1-demand_ig_p1,N=100)
uber_demand=running_mean(demand_p2-demand_ig_p2,N=100)
lyft_demand_final=np.mean(lyft_demand[-bdd:])
uber_demand_final=np.mean(uber_demand[-bdd:])
lyft_demand_var = np.std(lyft_demand[-bdd:])
uber_demand_var = np.std(uber_demand[-bdd:])
var_=[lyft_demand_var, uber_demand_var]

lyft_avg_price=np.mean(x_sgd[-100:,0,:]-x_sgd_ig[-100:,0,:])
uber_avg_price=np.mean(x_sgd[-100:,1,:]-x_sgd_ig[-100:,1,:])
lyft_var_price=np.std(x_sgd[-100:,0,:]-x_sgd_ig[-100:,0,:])
uber_var_price=np.std(x_sgd[-100:,1,:]-x_sgd_ig[-100:,1,:])
_var=[lyft_var_price, uber_var_price]

fig, ax = plt.subplots(1, 3, figsize=(10, 7))

data=['Lyft Revenue', 'Uber Revenue']
data_=['Lyft Demand', 'Uber Demand']
x_pos = [i for i, _ in enumerate(data)]
x_pos_ = [i for i, _ in enumerate(data_)]

vals=[lyft_rev_final, uber_rev_final]
vals_=[lyft_demand_final, uber_demand_final]
_vals=[lyft_avg_price,uber_avg_price]

dic={key: val for key, val in zip(data,vals)}
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

ax[0].set_ylabel("Change in Revenue", fontsize=fs-2)
ax[1].set_ylabel("Change in Demand", fontsize=fs-2)
ax[2].set_ylabel("Avg. Change in Price", fontsize=fs-2)
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Lyft','Uber'], fontsize=fs-2)
ax[0].bar([0,1], [val for key, val in sorted(dic.items())], yerr=var, color=['xkcd:hot pink', 'xkcd:slate grey'],error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'))
ax[1].bar(x_pos_, vals_,yerr=var_, color=['xkcd:hot pink', 'xkcd:slate grey'],error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'))
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['Lyft','Uber'], fontsize=fs-2)

ax[2].bar(x_pos_, _vals,yerr=_var, color=['xkcd:hot pink', 'xkcd:slate grey'],error_kw=dict(lw=5, capsize=5, capthick=3, color='xkcd:light pink'))
ax[2].set_xticks([0,1])
ax[2].set_xticklabels(['Lyft','Uber'], fontsize=fs-2)

plt.tick_params(labelsize=fs-2)
ax[0].tick_params(labelsize=fs-2)
ax[2].tick_params(labelsize=fs-2)
ax[1].tick_params(labelsize=fs-2)
plt.tight_layout()


plt.savefig(filename+'pdf', dpi=300, bbox_inches='tight', transparent=True)
    
## Plot change in revenue over time

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
lyft_rev=running_mean(rev_p1-rev_ig_p1,N=100)
uber_rev=running_mean(rev_p2-rev_ig_p2,N=100)
lyft_rev_final=np.mean(lyft_rev[-100:])
uber_rev_final=np.mean(uber_rev[-100:])

lyft_demand=running_mean(demand_p1-demand_ig_p1,N=100)
uber_demand=running_mean(demand_p2-demand_ig_p2,N=100)
lyft_demand_final=np.mean(lyft_demand[-100:])
uber_demand_final=np.mean(uber_demand[-100:])
fs=24
op=[1.0,0.5]
plt.plot(lyft_rev, color='xkcd:hot pink', linewidth=4, label='Lyft')
plt.plot(uber_rev, color='xkcd:slate grey', linewidth=4, label='Uber')
ax.grid(True)
ax.legend(fontsize=20,loc='upper right',ncol=2)
plt.tick_params(labelsize=20)
plt.xlabel('iterations', fontsize=fs)
plt.ylabel(r'change in revenue [$\Delta$ \$]', fontsize=fs)

fname="figs/test_end_files/price_change_price10_myopic."
df_all_sort_comp_l=df_all_l.sort_values(by=['price_change'], ascending=False)
df_all_sort_comp_u=df_all_u.sort_values(by=['price_change'], ascending=False)
fs_cb=22
import plotly.graph_objects as go
fig = go.Figure()
scale=10
f1=go.Scattermapbox(
    lat=df_all_sort_comp_u["centroid_lat"],
        lon=df_all_sort_comp_u["centroid_lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(#YlOrRd
            size=np.abs(df_all_sort_comp_u["price_change"])*scale, color=df_all_sort_comp_u["price_change"],colorscale= ["lightslategrey", "slategray", "black"],symbol = 'circle', opacity=0.9,
            showscale=True, 
cmax=5, #np.max([np.max(df_all_sort_comp_l["price_change"]),np.max(df_all_sort_comp_u["price_change"])])+5,
            cmin=0*np.min([np.min(df_all_sort_comp_l["price_change"]),np.min(df_all_sort_comp_u["price_change"])]),colorbar=dict(thickness=20,
                           ticklen=3, tickcolor='black',
                           tickfont=dict(size=fs_cb, color='black'))
        ),
        text="Uber",name="Uber"
    
        
    )
f2=go.Scattermapbox(
    lat=df_all_sort_comp_l["centroid_lat"],
        lon=df_all_sort_comp_l["centroid_lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=np.abs(df_all_sort_comp_l["price_change"])*scale, color=df_all_sort_comp_l["price_change"],colorscale= 'Burg',symbol = 'circle', opacity=0.9,
            showscale=True,
cmax=5, #np.max([np.max(df_all_sort_comp_l["price_change"]),np.max(df_all_sort_comp_u["price_change"])]),
cmin=0*np.min([np.min(df_all_sort_comp_l["price_change"]),np.min(df_all_sort_comp_u["price_change"])]),colorbar=dict(thickness=20,
                           ticklen=3, tickcolor='black',
                           tickfont=dict(size=fs_cb, color='black'))
        ),
        text="Lyft",
    name="Lyft"
        
    )
fig.add_trace(f1)
f2.marker.colorbar.x = 1.1 # Here
fig.add_trace(f2)
fig.update_layout(coloraxis_colorbar_x=-0.15)


fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken="pk.eyJ1IjoicmF0bGlmZmxqIiwiYSI6ImNqOGJ4cm8wcjAzN3QyeG1zcnZvMjB5bGUifQ.iRkpBPE-WANBkVc9ffI8ng",
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=df_all_sort_comp_u["centroid_lat"][0]-0.0175,
            lon=df_all_sort_comp_u["centroid_lon"][0]-0.021
        ),
        pitch=50,
        zoom=13.75
    ),

)

fig.update_geos(fitbounds="locations", resolution=110,)
fig.update_layout(height=600, width=1200,margin={"r":0,"t":0,"l":0,"b":0},legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",title_font_size=30, title="",
    x=0.01, orientation="v",bgcolor="white",font=dict( #"rgba(0,0,0,0)"
            size=30,
        ),
))

fig.write_image(fname+'pdf')

fig.show()

fname="figs/test_end_files/demand_change_price10_myopic."
df_all_sort_comp_l=df_all_l.sort_values(by=['demand_change'], ascending=False)
df_all_sort_comp_u=df_all_u.sort_values(by=['demand_change'], ascending=False)
fs_cb=22
import plotly.graph_objects as go
fig = go.Figure()
scale=0.75
f1=go.Scattermapbox(
    lat=df_all_sort_comp_u["centroid_lat"],
        lon=df_all_sort_comp_u["centroid_lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(#YlOrRd
            size=np.abs(df_all_sort_comp_u["demand_change"])*scale, color=df_all_sort_comp_u["demand_change"],colorscale= ["lightslategrey", "slategray", "black"],symbol = 'circle', opacity=0.9,
            showscale=True, 
cmax=np.max([np.max(df_all_sort_comp_l["demand_change"]),np.max(df_all_sort_comp_u["demand_change"])]),
            cmin=np.min([np.min(df_all_sort_comp_l["demand_change"]),np.min(df_all_sort_comp_u["demand_change"])]),colorbar=dict(thickness=20,
                           ticklen=3, tickcolor='black',
                           tickfont=dict(size=fs_cb, color='black'))
        ),
        text="Uber",name="Uber"
    
        
    )
f2=go.Scattermapbox(
    lat=df_all_sort_comp_l["centroid_lat"],
        lon=df_all_sort_comp_l["centroid_lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=np.abs(df_all_sort_comp_l["demand_change"])*scale, color=df_all_sort_comp_l["demand_change"],colorscale= 'Burg',symbol = 'circle', opacity=0.9,
            showscale=True,
cmax=np.max([np.max(df_all_sort_comp_l["demand_change"]),np.max(df_all_sort_comp_u["demand_change"])]),
cmin=np.min([np.min(df_all_sort_comp_l["demand_change"]),np.min(df_all_sort_comp_u["demand_change"])]),colorbar=dict(thickness=20,
                           ticklen=3, tickcolor='black',
                           tickfont=dict(size=fs_cb, color='black'))
        ),
        text="Lyft",
    name="Lyft"
        
    )
fig.add_trace(f1)
f2.marker.colorbar.x = 1.1 # Here
fig.add_trace(f2)
fig.update_layout(coloraxis_colorbar_x=-0.15)


fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken="pk.eyJ1IjoicmF0bGlmZmxqIiwiYSI6ImNqOGJ4cm8wcjAzN3QyeG1zcnZvMjB5bGUifQ.iRkpBPE-WANBkVc9ffI8ng",
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=df_all_sort_comp_u["centroid_lat"][0]-0.0175,
            lon=df_all_sort_comp_u["centroid_lon"][0]-0.021
        ),
        pitch=50,
        zoom=13.75
    ),

)

fig.update_geos(fitbounds="locations", resolution=110,)
fig.update_layout(height=600, width=1200,margin={"r":0,"t":0,"l":0,"b":0},legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",title_font_size=30, title="",
    x=0.01, orientation="v",bgcolor="white",font=dict( #"rgba(0,0,0,0)"
            size=30,
        ),
))
   

fig.write_image(fname+'pdf')

fig.show()

fname="figs/test_end_files/revenue_change_price10_myopic."
df_all_sort_comp_l=df_all_l.sort_values(by=['revenue_change'], ascending=False)
df_all_sort_comp_u=df_all_u.sort_values(by=['revenue_change'], ascending=False)
uberscale=["lightslategrey", "slategray", "black"]
import plotly.graph_objects as go
fig = go.Figure()
scale=2
f1=go.Scattermapbox(
        lat=df_all_sort_comp_u["centroid_lat"],
        lon=df_all_sort_comp_u["centroid_lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(#YlOrRd
            size=np.abs(df_all_sort_comp_u["revenue_change"])*scale, color=df_all_sort_comp_u["revenue_change"],colorscale= uberscale,symbol = 'circle', opacity=0.9,
            showscale=True, 
cmax=np.max([np.max(df_all_sort_comp_l["revenue_change"]),np.max(df_all_sort_comp_u["revenue_change"])]),
cmin=np.min([np.min(df_all_sort_comp_l["revenue_change"]),np.min(df_all_sort_comp_u["revenue_change"])]),colorbar=dict(thickness=20,
                           ticklen=2, tickcolor='black',
                           tickfont=dict(size=22, color='black'))
        ),
        text="Uber",name="Uber"
    )
f2=go.Scattermapbox(
        lat=df_all_sort_comp_l["centroid_lat"],
        lon=df_all_sort_comp_l["centroid_lon"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=np.abs(df_all_sort_comp_l["revenue_change"])*scale, color=df_all_sort_comp_l["revenue_change"],colorscale= 'Burg',symbol = 'circle', opacity=0.9,
            showscale=True,
cmax=np.max([np.max(df_all_sort_comp_l["revenue_change"]),np.max(df_all_sort_comp_u["revenue_change"])]),
cmin=np.min([np.min(df_all_sort_comp_l["revenue_change"]),np.min(df_all_sort_comp_u["revenue_change"])]),colorbar=dict(thickness=20,
                           ticklen=2, tickcolor='black',
                           tickfont=dict(size=22, color='black'))
        ),
        text="Lyft",
    name="Lyft"
    )
fig.add_trace(f1)
f2.marker.colorbar.x = 1.12 # Here
fig.add_trace(f2)
fig.update_layout(coloraxis_colorbar_x=-0.2)

fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken="pk.eyJ1IjoicmF0bGlmZmxqIiwiYSI6ImNqOGJ4cm8wcjAzN3QyeG1zcnZvMjB5bGUifQ.iRkpBPE-WANBkVc9ffI8ng",
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=df_all_sort_comp_u["centroid_lat"][0]-0.0175,
            lon=df_all_sort_comp_u["centroid_lon"][0]-0.021
        ),
        pitch=50,
        zoom=13.75
    )
)

fig.update_geos(fitbounds="locations", resolution=110,)
fig.update_layout(height=600, width=1200,margin={"r":0,"t":0,"l":0,"b":0},legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",title_font_size=30, title="",
    x=0.01, orientation="v",bgcolor="white",font=dict( #"rgba(0,0,0,0)"
            size=30,
        ),
))

fig.write_image(fname+'pdf')

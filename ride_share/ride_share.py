import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse
import pickle 
import sys,os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'../utils')
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


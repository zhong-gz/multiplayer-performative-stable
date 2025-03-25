import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from functions import est_varepsilon,data_distribution_map1,data_distribution_map2,remove_outliers_iqr,linear_data_generation
import copy
from datetime import datetime
import random
import time

def method_1(X,y,num_iters = 25,d_list = [10,1000,10000],map = 1,\
             strat_features = None,num_experiments = 10,seed_value = 42,alpha_1 = 2.1):

    # X = np.c_[np.ones((X.shape[0], 1)), X]
    method_name = 'PPW-AVG'
    num_d  = len(d_list)

    n = X.shape[0]
    model_int = Ridge(alpha = 1)#, fit_intercept=False)
    model_int.fit(X, y)
    print('Method 1:')
    model_list         = [[[model_int] for _ in range(num_d)] for _ in range(num_experiments)]
    model_gaps         = np.zeros((num_experiments, num_d, num_iters)) 
    mse_list_start     = np.zeros((num_experiments, num_d, num_iters))
    mse_list_end       = np.zeros((num_experiments, num_d, num_iters))

    for i in range(num_experiments):
        print('  Running {} th times'.format(i))
        np.random.seed(seed_value + i)
        random.seed(seed_value  + i)

        for k, d in enumerate(d_list):
            # initial model
            ridge_model = copy.deepcopy(model_int)
            norm_w_w = []
            varepsilon = []
            varepsilon_temp = 1
            X_old = np.copy(X)
            y_old = np.copy(y)

            print('     Running epsilon =  {}'.format(d))
            
            for t in range(num_iters):
                print(f'       Current iteration =  {t+1}, there are still {num_iters - t -1} iterations left', end='\r')
                # adjust distribution to current theta
                if map == 1:
                    X,y = linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = ridge_model, strat_features = strat_features)
                    
                if map == 2:
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = ridge_model)
                    
                # evaluate initial loss on the current distribution
                hat_y = ridge_model.predict(X_strat)
                mse = np.sqrt(mean_squared_error(y_strat, hat_y))
                mse_list_start[i,k,t] = mse

                # # learn on induced distribution
                gamma = max(1,alpha_1*varepsilon_temp)
                ridge_model_new = Ridge(alpha = gamma/2)
                ridge_model_new.fit(X_strat, y_strat)

                # evaluate final loss on the current distribution
                hat_y_new = ridge_model_new.predict(X_strat)
                mse = np.sqrt(mean_squared_error(y_strat, hat_y_new))
                mse_list_end[i,k,t] = mse

                # keep track of statistics
                model_list[i][k].append(ridge_model_new)
                varepsilon_star,norm_w_w = est_varepsilon(X_old,y_old,X_strat,y_strat,model_list[i][k],norm_w_w)
                varepsilon.append(varepsilon_star)
                varepsilon_no_outlier = remove_outliers_iqr(varepsilon)
                varepsilon_temp = np.max(varepsilon_no_outlier) 

                model_gaps[i,k,t] = np.linalg.norm(ridge_model_new.coef_-ridge_model.coef_)

                X_old = np.copy(X_strat)
                y_old = np.copy(y_strat)
                ridge_model = copy.deepcopy(ridge_model_new)
            print('')
            print('       gamma = ',gamma,'\n')
        print('-'*50)

    for k, d in enumerate(d_list):
        model_gaps_avg = np.nanmean(model_gaps, axis=0)
        model_gaps_std = np.std(model_gaps, axis=0)
        mse_list_start_avg = np.nanmean(mse_list_start, axis=0)
        mse_list_start_std = np.std(mse_list_start, axis=0)
        mse_list_end_avg = np.nanmean(mse_list_end, axis=0)
        mse_list_end_std = np.std(mse_list_end, axis=0)

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(method_name," Completion Time: ", current_time_str)

    return model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name

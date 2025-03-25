import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# D(w) = X - mu * w
def data_distribution_map1(X,y, mu = 0, model = None, strat_features = None):
    if model is not None:
        X_strat = X + mu * model.coef_ + np.random.normal(0, 0.1, size=model.coef_.shape)
        y_strat = y + mu * X @ (model.coef_.T**2) + np.random.normal(0, 0.1, size=y.shape)
        if np.any(y_strat > 2):
            # y_strat = np.clip(y_strat, -10, 10)
            scaler = MinMaxScaler()
            y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*2
    else:
        X_strat = np.copy(X)
        y_strat = np.copy(y)
    return X_strat,y_strat

# D(w) in crime rate prediction
def data_distribution_map2(X,y, mu = 0, model = None):
    # scaler = StandardScaler()
    # y_strat = np.zeros_like(y)
    # hat_y = model.predict(X)
    # scaled_hat_y = scaler.fit_transform(hat_y)
    # mean_scaled_hat_y = np.mean(scaled_hat_y)
    # y_strat = y - mu*(scaled_hat_y-mean_scaled_hat_y) + np.random.normal(0, 0.1, size=y.shape)

    scaler = MinMaxScaler()
    y_strat = np.zeros_like(y)
    hat_y = model.predict(X)
    mean_hat_y = np.mean(hat_y)
    y_strat = y - mu*(hat_y-mean_hat_y) + np.random.normal(0, 0.1, size=y.shape)
    if np.any(y_strat > 2):
        # y_strat = np.clip(y_strat, -10, 10)
        scaler = MinMaxScaler()
        y_strat = scaler.fit_transform(y_strat.reshape(-1, 1))*2

    # y_strat = np.clip(y - mu*(hat_y-mean_hat_y), a_min=0, a_max=1.2) + np.random.normal(0, 0.1, size=y.shape)
    # hat_y = model.predict(X)
    # scaler = StandardScaler()
    # scaled_hat_y = scaler.fit_transform(hat_y)
    # y_strat = (1 / (1 + np.exp(-(y - mu*(hat_y-scaled_hat_y))))) + np.random.normal(0, 0.1, size=y.shape)
    
    return X,y_strat

def linear_data_generation(n = 100,n_features = 20):
    X = np.random.rand(n, n_features)
    true_coefficients = [0.8, 0.5, 0.5, 0.3, 0.6, 0.3, 0.6, 0.2, 0.2, 0.4, 0.4, 0.3, 0.1, 0.0, 0.0, 0.3, 0.5, 0.7, 0.5, 0.9]
    y = X @ true_coefficients + np.random.randn(n) * 0.1
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    # scaler = MinMaxScaler()
    # y = scaler.fit_transform(y.reshape(-1, 1))
    return X,y

def linear_data_generation_sensitive(n = 100,n_features = 20,true_coefficients = None):
    X = np.random.rand(n, n_features)
    y = X @ true_coefficients + np.random.randn(n) * 0.1
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X,y

def est_varepsilon(X,y,X_new,y_new,w_arr,norm_w_w):
    ridge_model = w_arr[-1]
    
    # gradient of x
    n = X.shape[0]
    y_pred = ridge_model.predict(X)
    gradient = - (X.T.dot(y - y_pred)) # + 2*ridge_model.alpha * ridge_model.coef_.T
    mean_value = gradient #/n

    # gradient of x_new
    n_new = X_new.shape[0]
    y_pred_new = ridge_model.predict(X_new)
    gradient_new = - (X_new.T.dot(y_new - y_pred_new)) # + 2*ridge_model.alpha * ridge_model.coef_.T
    mean_value_new = gradient_new #/n_new

    if len(w_arr) == 2:
        norm_w_w.append(np.linalg.norm(w_arr[-1].coef_))
    else:
        norm_w_w.append(np.linalg.norm(w_arr[-1].coef_-w_arr[-2].coef_))

    if norm_w_w[-1] == 0:
        norm_w_w[-1] = 1e-4

    est_epsilon = np.linalg.norm(mean_value-mean_value_new)/(norm_w_w[-1])
    if est_epsilon < 1e-4:
        est_epsilon = 1e-4

    return est_epsilon,norm_w_w

def calculate_max_norm(data):
    max_norm = 0
    for d in data:
        norm = np.linalg.norm(d)
        if norm > max_norm:
            max_norm = norm
    return max_norm*max_norm

def remove_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data if x >= lower_bound and x <= upper_bound]
    return filtered_data

def preprocess_data_shift(X_strat, X, strat_features, n):
    if strat_features is None:
        strat_features = np.arange(X.shape[1])
    X_temp = np.copy(X_strat)
    X_combined = np.concatenate((X_temp, X), axis=0)
    X_subset = X_combined[:, strat_features]
    X_subset_scaled = preprocessing.scale(X_subset)
    X_combined[:, strat_features] = X_subset_scaled
    X_temp = X_combined[:n, :]
    return X_temp

def plot_step(i, offset, start_list, end_list, method_name, colors,markers,num_iters,c,linewidth = 1):
    if i == 1:
        plt.plot([i, i+offset], [start_list[c,i], end_list[c,i]],color=colors, marker=markers, linestyle='-', label=method_name)
    else:
        plt.plot([i, i+offset], [start_list[c,i], end_list[c,i]], color=colors, marker=markers, linestyle='-')
    if i < num_iters-1:
        plt.plot([i+offset, i+1], [end_list[c,i], start_list[c,i+1]], 'g:')

def plot_mse(mse_list_start,mse_list_start_std,colors,markers,linestyles,method_name,std=1,linewidth = 1):
    if method_name == 'PPW-AVG':
        plt.plot(range(len(mse_list_start)),mse_list_start,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=9,alpha=1)
    else:
        plt.plot(range(len(mse_list_start)),mse_list_start,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=6,alpha=1)
    if std == 1:
        plt.fill_between(range(len(mse_list_start)), mse_list_start - mse_list_start_std, mse_list_start + mse_list_start_std, color= colors, alpha=0.2, linewidth=0)

def plot_model_gap(model_gaps_avg,model_gaps_std,colors,markers,linestyles,method_name,std=1,linewidth = 1):
    if method_name == 'PPW-AVG':
        plt.plot(range(len(model_gaps_avg)),model_gaps_avg,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=9,alpha=1)
    else:
        plt.plot(range(len(model_gaps_avg)),model_gaps_avg,color=colors, marker=markers, linestyle=linestyles,label=method_name, linewidth=linewidth, markersize=6,alpha=1)
    if std == 1:
        plt.fill_between(range(len(model_gaps_avg)), model_gaps_avg - model_gaps_std, model_gaps_avg + model_gaps_std, alpha=0.2,color=colors, linewidth=0)

class CustomLogisticRegression(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, random_state=None, 
                 solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, 
                 warm_start=False, n_jobs=None, l1_ratio=None):
        super(CustomLogisticRegression, self).__init__(
            penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, 
            intercept_scaling=intercept_scaling, class_weight=class_weight, 
            random_state=random_state, solver=solver, max_iter=max_iter, 
            multi_class=multi_class, verbose=verbose, warm_start=warm_start, 
            n_jobs=n_jobs, l1_ratio=l1_ratio)
        
    def fit(self, X, y):
        super(CustomLogisticRegression, self).fit(X, y)
        self.w = self.coef_.ravel()

    def predict(self, X):
        w = self.coef_.ravel()
        b = self.intercept_
        score = np.dot(w,X.T) + b
        # score = self.sigmoid(z)
        # proba = self.predict_proba(X)[:,1]
        predictions = super(CustomLogisticRegression, self).predict(X)
        return score, predictions
    
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
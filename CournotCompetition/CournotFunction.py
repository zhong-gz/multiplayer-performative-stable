import pandas as pd
import numpy as np

def read_data(path = 'CournotCompetition/Global Crude Petroleum Trade 1995-2021.csv'):
    df = pd.read_csv(path)
    df_2021 = df[df['Year'] == 2021]
    trade_pivot = df_2021.pivot_table(
        index='Country', 
        columns='Action', 
        values='Trade Value', 
        aggfunc='sum', 
        fill_value=0
    )
    export_gt_import = trade_pivot[trade_pivot['Export'] > (trade_pivot['Import']+68.22*10000)].copy()
    result = pd.merge(
        export_gt_import.reset_index(),
        df_2021[['Country', 'Continent']].drop_duplicates(),
        on='Country',
        how='left'
    )
    export_gt_import['Net Export'] = export_gt_import['Export'] - export_gt_import['Import']
    export_gt_import['Crude Oil Volume (Barrels)'] = export_gt_import['Net Export'] / 68.22

    # 创建只包含国家和原油数量的新数据框
    global_oil_volume = pd.DataFrame({
        'Country': export_gt_import.index,
        'Crude Oil Volume (Barrels)': export_gt_import['Crude Oil Volume (Barrels)']
    })
    global_oil_volume['Crude Oil Volume (Barrels)'] = global_oil_volume['Crude Oil Volume (Barrels)'].astype(np.int64)
    # print("2021年出口额大于进口额的全球国家原油净出口量（桶）:")
    # print(global_oil_volume)
    return global_oil_volume

def distribution_map(z0,X,mu = 0.25):
    z = z0 + mu* np.arcsinh(1+ sum(X)) + np.random.normal(0, np.sqrt(0.1))
    return z

def runSIRR(z0,data,c,b,c_alg, MAXITER,mu):
    q = data
    total_q = np.sum(q)
    n = np.size(q)
    gamma = 1
    X_rr= np.empty((n, MAXITER+2))
    X_rr[:, 0] = 0
    z = np.empty(MAXITER+2)
    z[0] = z0
    eps = 0
    for i in range(MAXITER+1):
        gamma = max(0,eps * np.sqrt(n) * c_alg - b)
        A_mat = np.full((n, n), b)
        np.fill_diagonal(A_mat, A_mat.diagonal() + b + gamma)
        b_vec = -b * q - b * total_q - c + z[i]
        b_vec = b_vec.astype(np.float64)
        X_rr[:,i+1] = np.linalg.solve(A_mat, b_vec)

        z[i+1] = distribution_map(z0, X_rr[:,i+1], mu)

        eps = max(eps,(z[i+1] - z[i])/(np.linalg.norm(X_rr[:, i+1] - X_rr[:, i])+1e-3))

    dic={}
    dic['x']=X_rr[1:-1, :]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0) * (z[1:-1] - np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0))

    return dic

def runRR(z0,data,c,b, MAXITER,mu):
    q = data
    total_q = np.sum(q)
    n = np.size(q)
    gamma = 0.1
    X_rr= np.empty((n, MAXITER+2))
    X_rr[:, 0] = 0
    z = np.empty(MAXITER+2)
    z[0] = z0
    for i in range(MAXITER+1):
        A_mat = np.full((n, n), b)
        np.fill_diagonal(A_mat, A_mat.diagonal() + b + gamma)
        b_vec = -b * q - b * total_q - c + z[i]
        b_vec = b_vec.astype(np.float64)
        X_rr[:,i+1] = np.linalg.solve(A_mat, b_vec)
        z[i+1] = distribution_map(z0, X_rr[:,i+1], mu)

    dic={}
    dic['x']=X_rr[1:-1, :]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0) * (z[1:-1] - np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0))

    return dic

def runRGD(z0,data,c,b, MAXITER,mu,eta,x0):
    q = data
    total_q = np.sum(q)
    n = np.size(q)
    gamma = 0.1
    X_rg= np.empty((n, MAXITER+2))
    X_rg[:, 0] = x0
    z = np.empty(MAXITER+2)
    z[0] = z0
    for i in range(MAXITER+1):
        grad = b * (q + X_rg[:, i]) + b* np.sum(q[:, np.newaxis]+X_rg[:, i], axis=0) + c - z[i]
        X_rg[:, i+1] = X_rg[:, i] - eta * grad
        z[i+1] = distribution_map(z0, X_rg[:,i+1], mu)

    dic={}
    dic['x']=X_rg[1:-1, :]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rg[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0) * (z[1:-1] - np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0))

    return dic

def runSFB(z0,data,c,b, MAXITER,mu,eta,x0):
    q = data
    total_q = np.sum(q)
    n = np.size(q)
    gamma = 0.1
    X_rg= np.empty((n, MAXITER+2))
    X_rg[:, 0] = x0
    z = np.empty(MAXITER+2)
    z[0] = z0
    for i in range(MAXITER+1):
        grad = b * (q + X_rg[:, i]) + b* np.sum(q[:, np.newaxis]+X_rg[:, i], axis=0) + c - z[i]
        X_rg[:, i+1] = X_rg[:, i] - (eta**(-3/4)) * grad
        z[i+1] = distribution_map(z0, X_rg[:,i+1], mu)

    dic={}
    dic['x']=X_rg[1:-1, :]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rg[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0) * (z[1:-1] - np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0))

    return dic
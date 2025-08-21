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

def distribution_map(z0,X,mu = 0.25,b=1e-9):
    z = z0 - mu* np.arcsinh(1+ sum(X)) + np.random.normal(0, np.sqrt(0.0001))
    return z

def runSIRR(z0,data,c,b,c_alg, MAXITER,mu):
    q = data
    total_q = np.sum(q)
    n = np.size(q)
    gamma = 0.1
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

        z[i+1] = distribution_map(z0, X_rr[:,i+1], mu,b)

        if np.linalg.norm(X_rr[:, i+1] - X_rr[:, i]) > n*1e3: # average adjustment larger than 1e3 barrel than we consider it as adjustment rather than noise
            eps = max(eps,np.abs(z[i+1] - z[i])/np.linalg.norm(X_rr[:, i+1] - X_rr[:, i]))

    dic={}
    dic['x']=X_rr[:, 1:-1]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0)
    dic['revenue_total']=  np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0) * (z[1:-1] - b*np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0)) - c * np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0)
    dic['price']= (z[1:-1] - b*np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0))

    return dic

def runRR(z0,data,c,b, MAXITER,mu):
    q = data
    total_q = np.sum(q)
    n = np.size(q)
    gamma = 0
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
        z[i+1] = distribution_map(z0, X_rr[:,i+1], mu,b)

    dic={}
    dic['x']=X_rr[:, 1:-1]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rr[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0) * (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_rr[:, 1:-1]), axis=0))

    return dic

def runRGD(z0,data,c,b, MAXITER,mu,eta,x0):
    q = data
    n = np.size(q)
    X_rg= np.empty((n, MAXITER+2))
    X_rg[:, 0] = x0
    z = np.empty(MAXITER+2)
    z[0] = z0
    for i in range(MAXITER+1):
        grad = b * (q + X_rg[:, i]) + b* np.sum(q+X_rg[:, i]) + c - z[i]
        X_rg[:, i+1] = X_rg[:, i] - eta * grad/b
        z[i+1] = distribution_map(z0, X_rg[:,i+1], mu,b)

    dic={}
    dic['x']=X_rg[:, 1:-1]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rg[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0) * (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0))

    return dic

def runSFB(z0,data,c,b, MAXITER,mu,eta,x0):
    q = data
    n = np.size(q)
    X_rg= np.empty((n, MAXITER+2))
    X_rg[:, 0] = x0
    z = np.empty(MAXITER+2)
    z[0] = z0
    for i in range(MAXITER+1):
        grad = b * (q + X_rg[:, i]) + b* np.sum(q+X_rg[:, i]) + c - z[i]
        X_rg[:, i+1] = X_rg[:, i] - 5*(eta*(i+1)**(-3/4)) * grad/b
        z[i+1] = distribution_map(z0, X_rg[:,i+1], mu,b)

    dic={}
    dic['x']=X_rg[:, 1:-1]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_rg[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0) * (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_rg[:, 1:-1]), axis=0))

    return dic

def runAGM(z0,data,c,b, MAXITER,mu,eta,x0):
    q = data
    n = np.size(q)
    X_ag= np.empty((n, MAXITER+2))
    X_ag[:, 0] = x0
    z = np.empty(MAXITER+2)
    z[0] = z0
    A = np.random.rand(1)*b
    for i in range(MAXITER+1):
        grad = b * (q + X_ag[:, i]) + b* np.sum(q+X_ag[:, i]) + c - z[i] - A *(q+ X_ag[:, i])
        X_ag[:, i+1] = X_ag[:, i] - (eta * grad/b)*0.4
        if i > 0:
            for j in range(10): #update A 10 times
                A = update_estimate(A,X_ag[:,i],z0,n, mu,b,z[i])
        z[i+1] = distribution_map(z0, X_ag[:,i+1], mu,b)

    dic={}
    dic['x']=X_ag[:, 1:-1]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_ag[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0) * (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0))

    return dic

def update_estimate(A,x,z0,n, mu,b,zt):
    '''
    least squares update
    '''
    nu= 1e-7 #1e-18 1e-5
    # query environment
    ut = np.random.normal(0,1e8,size=n)  #1e4
    q=distribution_map(z0, x+ut, mu,b)
    g = (q-zt-A*sum(ut))*sum(ut)
    power = int(np.floor(np.log10(np.max(np.abs(g)))))
    scale_factor = 10 ** power
    g = g/scale_factor
    
    Astar = A+ nu*g
    return Astar

def runOPGD(z0,data,c,b, MAXITER,mu,eta,x0):
    q = data
    n = np.size(q)
    X_ag= np.empty((n, MAXITER+2))
    X_ag[:, 0] = x0
    z = np.empty(MAXITER+2)
    z[0] = z0
    A = np.random.rand(1)*b
    for i in range(MAXITER+1):
        grad = b * (q + X_ag[:, i]) + b* np.sum(q+X_ag[:, i]) + c - z[i] - A *(q+ X_ag[:, i])
        X_ag[:, i+1] = X_ag[:, i] - (eta * grad/b)#*0.1
        if i > 0:
            for j in range(10): #update A 10 times
                A = update_estimate_OPGD(A,z0,n, mu,b)
        z[i+1] = distribution_map(z0, X_ag[:,i+1], mu,b)

    dic={}
    dic['x']=X_ag[:, 1:-1]
    dic['quantity_total']= np.sum(q[:, np.newaxis]+X_ag[:, 1:-1], axis=0)
    dic['revenue_total']= np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0) * (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0)) - c * np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0)
    dic['price']= (z[1:-1] - b*np.sum((q[:, np.newaxis]+X_ag[:, 1:-1]), axis=0))
    return dic

def update_estimate_OPGD(A,z0,n, mu,b):
    '''
    least squares update
    '''
    nu= 1e-9
    # query environment
    ut = np.random.normal(0,1,size=n)
    y=distribution_map(z0, ut, mu,b)
    g = (A*sum(ut)-y)*sum(ut)
    power = int(np.floor(np.log10(np.max(np.abs(g)))))
    scale_factor = 10 ** power
    g = g/scale_factor

    Astar = A- nu*g
    return Astar
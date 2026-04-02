# import sys
# print("== 当前 Python 环境 ==")
# print("解释器路径:", sys.executable)
# print("虚拟环境根目录:", sys.prefix)  # venv 环境会包含 "venv" 字样

import numpy as np
import pandas as pd
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
import random
from scipy.special import softmax

from scipy.sparse import random as scirand

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class ddstrategic_prediction:
    def __init__(self,lam=None, params=None, maxx=10, n=2, d=10, m=2, N_test=100, 
                 MAXITER=1000, sigma_theta=0.01, sigma_w=0.01, sigma_z1=0.01, sigma_z2=0.01, B=None,
                 seed=2, nu=1e-3, mu_w1=0, mu_w2=0, mu_theta=0):
        # Set defaults
        if lam is None:
            lam = [0.0] * n
        if params is None:
            params = {}
        if B is None:
            B = np.zeros((d, 1))
            
        self.n = n
        self.d = d
        self.m = m
        self.lam = lam  # List of regularization parameters for each player
        self.l = [-maxx for i in range(self.d)]
        self.u = [maxx for i in range(self.d)]
        N = 1000
        self.N_test = N_test
        self.MAXITER = MAXITER
        self.sigma_theta = sigma_theta
        self.sigma_w = sigma_w
        self.sigma_z1 = sigma_z2
        self.sigma_z2 = sigma_z1
        self.mu_w1 = mu_w1
        self.mu_w2 = mu_w2
        self.B = B
        
        # Store matrices as lists for each player
        self.A = []
        self.Ac = []
        self.C = []
        self.A_hat = []
        self.Ac_hat = []
        
        for i in range(n):
            # Get matrices for player i (fall back to default if not provided)
            a_key = f'A{i+1}' if f'A{i+1}' in params else f'A'
            ac_key = f'Ac{i+1}' if f'Ac{i+1}' in params else f'Ac'
            c_key = f'C{i+1}' if f'C{i+1}' in params else f'C'
            
            self.A.append(params.get(a_key, np.zeros((1, self.d))))
            self.Ac.append(params.get(ac_key, np.zeros((1, self.d))))
            self.C.append(params.get(c_key, np.zeros((self.d, self.d))))
            
            # Initialize estimates
            self.A_hat.append([np.zeros((1, self.d))])
            self.Ac_hat.append([np.zeros((1, self.d))])
        
        self.seed = seed
        np.random.seed(seed)
        self.nu = nu
        self.mu_theta = mu_theta

    def distribution_map(self, x, theta):
        """Map state to distribution for n players"""
        z = []
        for i in range(self.n):
            # Compute interaction term: sum of Ac_i @ x_j for j != i
            interaction = np.zeros((self.m,))
            for j in range(self.n):
                if i != j:
                    interaction += self.Ac[i] @ x[j]
            
            z_i = self.D_w(i) + self.A[i] @ x[i] + interaction + theta.T @ self.B.flatten()
            z.append(z_i)
        
        # Update theta
        u_i = np.random.normal(0, self.sigma_w, size=(self.d,))
        theta_sum = u_i.copy()
        for i in range(self.n):
            theta_sum += self.C[i] @ x[i]
        thetaT = theta_sum + theta.T
        
        return z, thetaT.T

    def D_theta(self):
        return np.random.normal(self.mu_theta,self.sigma_theta,size=(self.d,self.m))

    def D_w(self, player):
        """Generate random noise for player"""
        return np.random.normal(self.mu_w1, self.sigma_w, size=(self.m,))

    def getgrad_so(self,x,theta):
        w=self.D_w(0)
        p1=((self.A1-theta.T).T@(theta.T@self.B.flatten()+self.A1@x[0]+self.Ac1@x[1]+w-theta.T@x[0])+self.lam1*x[0]
            +self.Ac2.T@(theta.T@self.B.flatten()+self.A2@x[1]+self.Ac2@x[0]+w-theta.T@x[1]))

        w=self.D_w(1)
        p2=((self.A2-theta.T).T@(theta.T@self.B.flatten()+self.A2@x[1]+self.Ac2@x[0]+w-theta.T@x[1])+self.lam2*x[1]
            +self.Ac1.T@(theta.T@self.B.flatten()+self.A1@x[0]+self.Ac1@x[1]+w-theta.T@x[0]))

        return np.vstack((p1.T,p2.T))

    def getgrad(self, x, theta):
        """Compute gradient for all players"""
        z, theta_new = self.distribution_map(x, theta)
        grads = []
        
        for i in range(self.n):
            w = self.D_w(i)
            # Compute interaction term
            interaction_grad = np.zeros(self.d)
            for j in range(self.n):
                if i != j:
                    interaction_grad += self.Ac[i] @ x[j]
            
            p_i = (self.A[i] - theta_new.T).T @ (
                theta_new.T @ self.B.flatten() + 
                self.A[i] @ x[i] + 
                interaction_grad + w - 
                theta_new.T @ x[i]
            ) / self.m + self.lam[i] * x[i]
            
            grads.append(p_i)
        
        return np.vstack([g.T for g in grads])

    def getHess(self,x,th):
        """Compute Hessian for all players"""
        Hessians = []
        for i in range(self.n):
            H_i = (self.A[i]-th.T).T@(self.A[i]-th.T) + self.lam[i]*np.eye(self.d)
            Hessians.append(H_i)
        return Hessians

    def getgrad_agd(self, x, theta, Ahats=[], AChats=[], passvals=False):
        """AGD gradient for all players using estimates"""
        if not passvals:
            Ahats = [self.A_hat[i][-1] if i < len(self.A_hat) else self.A[i] for i in range(self.n)]
            AChats = [self.Ac_hat[i][-1] if i < len(self.Ac_hat) else self.Ac[i] for i in range(self.n)]
        
        grads = []
        for i in range(self.n):
            w = self.D_w(i)
            # Compute sum of other players' actions: sum_{j!=i} Ac_hat[i,j] @ x[j]
            interaction_term = np.zeros(self.d)
            for j in range(self.n):
                if i != j and j < len(AChats):
                    interaction_term += AChats[i] @ x[j]
            
            p_i = (Ahats[i] - theta.T).T @ (theta.T @ self.B.flatten() + Ahats[i] @ x[i] + interaction_term + w - theta.T @ x[i]) / self.m + self.lam[i] * x[i]
            grads.append(p_i)
        
        return np.vstack([g.T for g in grads])
    
    def getgrad_opgd(self, x, theta, Ahats_opgd=[]):
        """OPGD gradient for all players using expanded A_hat matrices"""
        z_list, theta1 = self.distribution_map(x, theta)
        
        if not Ahats_opgd:
            Ahats_opgd = [np.zeros((self.d+1, self.d)) for _ in range(self.n)]
        
        grads = []
        for i in range(self.n):
            w = self.D_w(i)
            A_hat_i = Ahats_opgd[i]
            # A_hat is (d+1) x d: [A_hat_diag; intercept]
            p_i = (A_hat_i[-1, :] - 2*A_hat_i[:-1, :] @ x[i] - theta.T).T @ ((A_hat_i[-1, :] @ x[i] + theta.T @ self.B.flatten() + w) - (A_hat_i[:-1, :] @ x[i] + theta.T) @ x[i])
            grads.append(p_i)
        
        return np.vstack([g.T for g in grads])
    
    def update_estimate_opgd(self, x, z_list, theta, v_t=1, Ahats_opgd=[]):
        """Update A_hat estimates for OPGD for all players"""
        if not Ahats_opgd:
            Ahats_opgd = [np.zeros((self.d+1, self.d)) for _ in range(self.n)]
        
        Ahats_opgd_new = []
        for i in range(self.n):
            u_i = np.random.normal(0, 1, size=(self.d,))
            x_u = list(x)
            x_u[i] = x_u[i] + u_i
            q_list, theta_tmp = self.distribution_map(x_u, theta)
            
            A_hat_i = Ahats_opgd[i]
            # Update A_hat_i
            update_term = (A_hat_i @ u_i).reshape(-1, 1) - np.vstack((theta_tmp, q_list[i].reshape(-1, 1)))
            A_hat_i_new = A_hat_i - v_t * update_term.mean(axis=1).reshape(-1, 1) @ u_i.reshape(1, -1)
            Ahats_opgd_new.append(A_hat_i_new)
        
        return Ahats_opgd_new

    def update_estimate(self, x, z_list, theta, nu=1e-3, mu=1, Ahats=[], AChats=[], UNCORR=False, passvals=False):
        """Update A_hat and Ac_hat estimates for all players"""
        if not passvals:
            Ahats = [self.A_hat[i][-1] if i < len(self.A_hat) else self.A[i].copy() for i in range(self.n)]
            AChats = [self.Ac_hat[i][-1] if i < len(self.Ac_hat) else self.Ac[i].copy() for i in range(self.n)]
        
        # Random perturbations for each player
        u_list = [np.random.normal(0, mu, size=(self.d,)) for _ in range(self.n)]
        x_u = [x[i] + u_list[i] for i in range(self.n)]
        q_list, theta_plus = self.distribution_map(x_u, theta)
        
        z_list_orig, theta_orig = self.distribution_map(x, theta)
        
        Ahats_new = []
        AChats_new = []
        
        for i in range(self.n):
            # Construct combined matrix [A_hat | Ac_hat] for player i
            barA_hat = np.hstack([Ahats[i], *[AChats[i_ac] for i_ac in range(i)] + [AChats[i]] + [AChats[i_ac] for i_ac in range(i+1, self.n)]])
            # Construct stacked perturbation vector
            u_stack = np.hstack([u_list[ii] for ii in range(self.n)]).reshape(-1, 1)
            
            if UNCORR:
                dA = np.diag(np.diagonal(Ahats[i]))
                barA_hat_diag = np.hstack([dA] + [np.diag(np.diagonal(AChats[i_ac])) for i_ac in range(self.n) if i_ac != i])
                barA_hat = barA_hat_diag
            
            # Update rule: A_hat_new = A_hat + nu * (q - z - A_hat @ u) @ u^T
            residual = q_list[i].reshape(-1, 1) - z_list_orig[i].reshape(-1, 1) - barA_hat @ u_stack
            barA_hat_new = barA_hat + nu * residual @ u_stack.T
            
            if UNCORR:
                A_hat_new = np.diag(np.diagonal(barA_hat_new[:, :self.d]))
                Ac_hat_list = []
                col_idx = self.d
                for i_ac in range(self.n):
                    if i_ac == i:
                        Ac_hat_list.append(np.diag(np.diagonal(barA_hat_new[:, col_idx:col_idx+self.d])))
                    else:
                        Ac_hat_list.append(np.diag(np.diagonal(barA_hat_new[:, col_idx:col_idx+self.d])))
                    col_idx += self.d
            else:
                A_hat_new = barA_hat_new[:, :self.d]
                Ac_hat_list = []
                col_idx = self.d
                for i_ac in range(self.n):
                    Ac_hat_list.append(barA_hat_new[:, col_idx:col_idx+self.d])
                    col_idx += self.d
            
            Ahats_new.append(A_hat_new)
            AChats_new.append(Ac_hat_list[i])
        
        return Ahats_new, AChats_new

    def proj(self,x):
        y=np.zeros(np.shape(x))
        for i in range(self.n):
            for j in range(self.d):
                if x[i][j]<=self.l[j]:
                    y[i][j]=self.l[j]
                elif self.l[j]<x[i][j] and x[i][j]<self.u[j]:
                    y[i][j]=x[i][j]
                else:
                    y[i][j]=self.u[j]
        return y

    def get_loss(self, x, z_list, theta):
        """Compute losses for all players"""
        losses = []
        for i in range(self.n):
            z_i = z_list[i] + self.A[i] @ x[i]
            # Add interaction terms
            for j in range(self.n):
                if i != j:
                    z_i += self.Ac[i] @ x[j]
            z_i += theta.T @ self.B.flatten()
            
            loss_i = 0.5 * z_i.T @ z_i + self.lam[i] * la.norm(x[i])
            losses.append(loss_i)
        
        return losses

    def getgrad_rgd(self, x, z_, theta):
        """RGD gradient for all players"""
        grads = []
        for i in range(self.n):
            p_i = -theta @ (z_[i] - theta.T @ x[i]) / self.m + self.lam[i] * x[i]
            grads.append(p_i)
        return np.vstack([g.T for g in grads])
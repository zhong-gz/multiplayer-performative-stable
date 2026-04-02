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
            # Get matrices for player i from params
            # params should have keys 'A', 'Ac', 'C' which are lists indexed by player i
            if 'A' in params and isinstance(params['A'], list):
                self.A.append(params['A'][i])
            else:
                self.A.append(params.get(f'A{i+1}', np.zeros((1, self.d))))
            
            if 'Ac' in params and isinstance(params['Ac'], list):
                self.Ac.append(params['Ac'][i])
            else:
                self.Ac.append(params.get(f'Ac{i+1}', np.zeros((1, self.d))))
            
            if 'C' in params and isinstance(params['C'], list):
                self.C.append(params['C'][i])
            else:
                self.C.append(params.get(f'C{i+1}', np.zeros((self.d, self.d))))
            
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
        # Get B contribution: B should be (d,) or (d, 1)
        if self.B.ndim == 1:
            B_vec = self.B
        elif self.B.ndim == 2:
            B_vec = self.B[:, 0]  # Take first column
        else:
            B_vec = self.B.flatten()
        
        for i in range(self.n):
            # Start with noise: shape (m,)
            z_i = self.D_w(i)
            
            # Add own action: A[i] @ x[i], where A[i] is (1,d), x[i] is (d,)
            # Result is (1,) - flatten to scalar and broadcast to (m,)
            z_i = z_i + (self.A[i] @ x[i]).flatten()[0]
            
            # Add cross-player interactions: sum of Ac[i] @ x[j] for j != i
            for j in range(self.n):
                if i != j:
                    # Ac[i] is (1,d), x[j] is (d,), result is (1,) - flatten to scalar
                    z_i = z_i + (self.Ac[i] @ x[j]).flatten()[0]
            
            # Add theta contribution: theta.T @ B_vec
            # theta is (d, m), B_vec is (d,), result is (m,)
            z_i = z_i + theta.T @ B_vec
            z.append(z_i)
        
        # Update theta
        theta_sum = np.zeros(self.d)
        for i in range(self.n):
            theta_sum = theta_sum + self.C[i] @ x[i]
        # thetaT is (d, m): sum of C @ x, then add original theta
        thetaT = theta_sum.reshape(-1, 1) + theta  # theta is (d, m)
        
        return z, thetaT

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
        
        # Get B vector
        if self.B.ndim == 1:
            B_vec = self.B
        elif self.B.ndim == 2:
            B_vec = self.B[:, 0]
        else:
            B_vec = self.B.flatten()
        
        grads = []
        
        for i in range(self.n):
            w = self.D_w(i)
            # Compute interaction term: sum of Ac[i] @ x[j] for j != i
            interaction_grad = 0.0
            for j in range(self.n):
                if i != j:
                    interaction_grad += (self.A[i] @ x[j]).flatten()[0] if self.A[i].ndim > 1 else self.A[i] @ x[j]
            
            # Compute components:
            # A[i] is (1, d) or (d,)
            # theta_new is (d, m)
            # x[i] is (d,)
            
            # Ensure A[i] and theta_new are properly formatted
            A_i = self.A[i].flatten()  # Convert to (d,)
            theta_T = theta_new.T  # This should be (m, d)
            
            # Compute signal contributions
            a_contrib = A_i @ x[i]
            b_contrib = theta_T @ B_vec  # (m, d) @ (d,) = (m,)
            if b_contrib.ndim > 0:
                b_contrib = b_contrib[0]  # Take first for scalar
            
            # Compute full signal
            signal = z[i] + a_contrib + interaction_grad + b_contrib  # (m,)
            
            # Compute gradient: (A[i] - theta_new.T).T @ signal / m + lambda * x[i]
            # A_i - theta_T: (d,) - (m, d) broadcasts to (m, d)
            # (A_i - theta_T).T: (d, m)
            grad_mat = (A_i - theta_T)  # (m, d) form
            p_i = (A_i - theta_T.mean(axis=0)) * signal.mean() / self.m + self.lam[i] * x[i]
            
            grads.append(p_i)
        
        return np.vstack([g for g in grads])

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
        
        # Get B vector
        if self.B.ndim == 1:
            B_vec = self.B
        elif self.B.ndim == 2:
            B_vec = self.B[:, 0]
        else:
            B_vec = self.B.flatten()
        
        grads = []
        for i in range(self.n):
            w = self.D_w(i)
            
            # Extract A matrices as vectors
            A_hat_i = Ahats[i].flatten()  # (1, d) -> (d,)
            theta_mean = theta.mean(axis=1)  # (d, m) -> (d,)
            
            # Compute signal: estimate of z_i at current state
            z_approx_scalar = (A_hat_i @ x[i])
            for j in range(self.n):
                z_approx_scalar += (AChats[i].flatten() @ x[j])
            
            # Gradient: (A_hat[i] - theta_mean) @ signal + lambda * x[i]
            p_i = (A_hat_i - theta_mean) * z_approx_scalar / self.m + self.lam[i] * x[i]
            grads.append(p_i)
        
        return np.vstack([g for g in grads])
    
    def getgrad_opgd(self, x, theta, Ahats_opgd=[]):
        """OPGD gradient for all players using expanded A_hat matrices"""
        z_list, theta1 = self.distribution_map(x, theta)
        
        if not Ahats_opgd:
            Ahats_opgd = [np.zeros((self.d+1, self.d)) for _ in range(self.n)]
        
        # Get B vector
        if self.B.ndim == 1:
            B_vec = self.B
        elif self.B.ndim == 2:
            B_vec = self.B[:, 0]
        else:
            B_vec = self.B.flatten()
        
        grads = []
        for i in range(self.n):
            w = self.D_w(i)
            A_hat_i = Ahats_opgd[i]
            
            # A_hat is (d+1) x d where last row is bias/intercept
            # Extract the linear part (first d rows) and bias (last row)
            A_mat = A_hat_i[:-1, :]  # (d, d)
            a_bias = A_hat_i[-1, :]  # (d,)
            
            # Prediction: A_mat @ x[i] + bias_vec (should give approximately z_list[i])
            # Note: z_list[i] comes from distribution_map and is shape (m,)
            # We use the mean signal for gradient computation
            
            # Simple OPGD gradient: use estimated vs actual
            # Gradient should be scalar field (d,) to match x[i]
            # For simplicity, use z_list[i].mean() as target
            z_target_scalar = z_list[i].mean()
            
            # Compute gradient as average direction
            A_mat_flat = A_mat.flatten()  # (d*d,)
            grad_signal = z_target_scalar  # scalar
            
            # Gradient update for x[i]
            p_i = (A_mat.T @ np.ones(self.d) - theta[:, :self.d].mean(axis=1) * grad_signal) / self.m + self.lam[i] * x[i]
            
            grads.append(p_i)
        
        return np.vstack([g for g in grads])
    
    def update_estimate_opgd(self, x, z_list, theta, v_t=1, Ahats_opgd=[]):
        """Update A_hat estimates for OPGD for all players"""
        if not Ahats_opgd:
            Ahats_opgd = [np.zeros((self.d+1, self.d)) for _ in range(self.n)]
        
        Ahats_opgd_new = []
        for i in range(self.n):
            u_i = np.random.normal(0, 1, size=(self.d,))
            x_u = [x[j].copy() if j != i else x[j] + u_i for j in range(self.n)]
            q_list, theta_tmp = self.distribution_map(x_u, theta)
            
            A_hat_i = Ahats_opgd[i]
            # Simple online update: A_hat tracks the signal z_list[i]
            # For each observation, update A_hat to reduce error
            
            # Use gradient descent to update A_hat
            # Target signal is z_list[i] (m,)
            # Use mean signal as target
            signal_mean = z_list[i].mean()
            
            # Simple update rule - just drift towards mean
            A_hat_i_new = A_hat_i - v_t * (A_hat_i.mean(axis=1, keepdims=True) - signal_mean / (self.d + 1))
            
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
        
        # Get z at current state
        z_list_orig, theta_orig = self.distribution_map(x, theta)
        
        Ahats_new = []
        AChats_new = []
        
        for i in range(self.n):
            # Construct combined [A_hat | Ac_hat] matrix
            A_mat = Ahats[i]  # (1, d)
            Ac_mats = []
            for j in range(self.n):
                if i != j:
                    Ac_mats.append(AChats[i])
            
            barA_hat = np.hstack([A_mat] + Ac_mats) if Ac_mats else A_mat
            
            # Stack perturbation vector
            u_stack = np.hstack(u_list).reshape(-1, 1)
            
            # Compute residual: q - z - A_hat @ u
            residual = q_list[i].reshape(-1, 1) - z_list_orig[i].reshape(-1, 1) - barA_hat @ u_stack
            
            # Update rule
            barA_hat_new = barA_hat + nu * residual.mean(axis=0, keepdims=True) @ u_stack.T
            
            # Extract A_hat and Ac_hat from updated combined matrix
            A_hat_new = barA_hat_new[:, :self.d]
            Ahats_new.append(A_hat_new)
            
            Ac_mats_new = []
            col_idx = self.d
            for j in range(self.n):
                if i != j:
                    Ac_mats_new.append(barA_hat_new[:, col_idx:col_idx+self.d])
                    col_idx += self.d
            
            if Ac_mats_new:
                AChats_new.append(np.mean(Ac_mats_new, axis=0))  # Average all interaction terms
            else:
                AChats_new.append(np.zeros_like(AChats[i]))
        
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
            # z_list[i] is (m,), should compute full loss over all samples
            z_full = z_list[i] + (self.A[i] @ x[i]).flatten()[0]
            for j in range(self.n):
                if i != j:
                    z_full = z_full + (self.Ac[i] @ x[j]).flatten()[0]
            z_full = z_full + theta.T @ self.B.flatten()
            
            # Loss is 0.5 * sum(z_full^2) + lambda * norm(x[i])
            loss_i = 0.5 * np.sum(z_full**2) + self.lam[i] * la.norm(x[i])
            losses.append(loss_i)
        
        return losses

    def getgrad_rgd(self, x, z_, theta):
        """RGD gradient for all players"""
        grads = []
        for i in range(self.n):
            # z_[i] is (m,), theta is (d, m), x[i] is (d,)
            # signal = z_[i] - theta.T @ x[i], shape (m,)
            # p_i = -theta @ signal / m + lambda * x[i]
            signal = z_[i] - theta.T @ x[i]
            p_i = -theta @ signal / self.m + self.lam[i] * x[i]
            grads.append(p_i)
        return np.vstack([g for g in grads])
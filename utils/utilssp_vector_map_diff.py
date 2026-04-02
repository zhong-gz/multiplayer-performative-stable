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
from joblib import Parallel, delayed

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
        """Map state to distribution for n players (vectorized + parallel)"""
        # Get B contribution: B should be (d,) or (d, 1)
        if self.B.ndim == 1:
            B_vec = self.B
        elif self.B.ndim == 2:
            B_vec = self.B[:, 0]  # Take first column
        else:
            B_vec = self.B.flatten()
        
        # Compute theta contribution once (shared across all players)
        theta_contrib = theta.T @ B_vec  # (m,)
        
        # Parallel computation of z_i for each player
        def compute_z_i(i):
            # Noise for player i
            z_i = self.D_w(i)
            
            # Own action contribution
            z_i = z_i + (self.A[i] @ x[i]).flatten()[0]
            
            # Cross-player interactions
            for j in range(self.n):
                if i != j:
                    z_i = z_i + (self.Ac[i] @ x[j]).flatten()[0]
            
            # Add theta contribution
            z_i = z_i + theta_contrib
            return z_i
        
        # Parallelize z computation across all players
        z = Parallel(n_jobs=-1)(delayed(compute_z_i)(i) for i in range(self.n))
        
        # Update theta: vectorized (not parallelized as it's small)
        theta_sum = np.zeros(self.d)
        for i in range(self.n):
            theta_sum = theta_sum + self.C[i] @ x[i]
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
        """Compute gradient for all players (vectorized + parallel)"""
        z, theta_new = self.distribution_map(x, theta)
        
        # Get B vector
        if self.B.ndim == 1:
            B_vec = self.B
        elif self.B.ndim == 2:
            B_vec = self.B[:, 0]
        else:
            B_vec = self.B.flatten()
        
        # Compute theta mean (shared)
        theta_T = theta_new.T  # (m, d)
        theta_mean = theta_T.mean(axis=0)  # (d,)
        
        # Parallel gradient computation for each player
        def compute_grad_i(i):
            A_i = self.A[i].flatten()  # (d,)
            # z[i] is (m,) - the signal for player i
            signal_mean = np.mean(z[i])  # scalar mean signal
            
            # Create gradient: (A[i] - theta_mean) * signal_mean / m + lambda * x[i]
            grad_contrib = (A_i - theta_mean) * signal_mean / self.m
            p_i = grad_contrib + self.lam[i] * x[i]
            return p_i
        
        # Parallelize gradient computation
        grads = Parallel(n_jobs=-1)(delayed(compute_grad_i)(i) for i in range(self.n))
        
        return np.vstack([g for g in grads])

    def getHess(self,x,th):
        """Compute Hessian for all players (vectorized + parallel)"""
        # Compute theta mean (shared)
        th_T = th.T  # (m, d)
        th_mean = th_T.mean(axis=0)  # (d,)
        
        # Parallel Hessian computation
        def compute_hess_i(i):
            A_i = self.A[i].flatten()  # (d,)
            # H_i = (A[i] - th_mean) @ (A[i] - th_mean).T + lambda * I
            diff = A_i - th_mean  # (d,)
            H_i = np.outer(diff, diff) + self.lam[i] * np.eye(self.d)
            return H_i
        
        # Parallelize Hessian computation
        Hessians = Parallel(n_jobs=-1)(delayed(compute_hess_i)(i) for i in range(self.n))
        
        return Hessians

    def getgrad_agd(self, x, theta, Ahats=[], AChats=[], passvals=False):
        """AGD gradient for all players using estimates (vectorized + parallel)"""
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
        
        # Compute theta mean (shared)
        theta_mean = theta.mean(axis=1)  # (d,)
        
        # Parallel AGD gradient computation
        def compute_agd_grad_i(i):
            A_hat_i = Ahats[i].flatten()  # (1, d) -> (d,)
            
            # Compute signal estimate: sum of estimated interactions
            z_approx = (A_hat_i @ x[i])
            for j in range(self.n):
                if i != j:
                    z_approx += (AChats[i].flatten() @ x[j])
            
            # Gradient: (A_hat[i] - theta_mean) * signal / m + lambda * x[i]
            p_i = (A_hat_i - theta_mean) * z_approx / self.m + self.lam[i] * x[i]
            return p_i
        
        # Parallelize AGD gradient computation
        grads = Parallel(n_jobs=-1)(delayed(compute_agd_grad_i)(i) for i in range(self.n))
        
        return np.vstack([g for g in grads])
    
    def getgrad_opgd(self, x, theta, Ahats_opgd=[]):
        """OPGD gradient for all players using expanded A_hat matrices (vectorized)"""
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
            A_hat_i = Ahats_opgd[i]
            
            # Extract linear part (first d rows) and bias (last row)
            A_mat = A_hat_i[:-1, :]  # (d, d)
            a_bias = A_hat_i[-1, :]  # (d,)
            
            # Use mean signal for gradient approximation
            z_target_scalar = np.mean(z_list[i])
            
            # Simplified OPGD gradient
            # Gradient for x[i]: mean column of A_mat influence * signal
            A_mean = A_mat.mean(axis=1)  # (d,)
            p_i = (A_mean - theta1[:, :self.d].mean(axis=1)) * z_target_scalar / self.m + self.lam[i] * x[i]
            
            grads.append(p_i)
        
        return np.vstack([g for g in grads])
    
    def update_estimate_opgd(self, x, z_list, theta, v_t=1, Ahats_opgd=[]):
        """Update A_hat estimates for OPGD for all players (vectorized)"""
        if not Ahats_opgd:
            Ahats_opgd = [np.zeros((self.d+1, self.d)) for _ in range(self.n)]
        
        # Vectorized perturbation for all players (compute all at once)
        u_list = [np.random.normal(0, 1, size=(self.d,)) for _ in range(self.n)]
        x_u = [x[i] + u_list[i] for i in range(self.n)]
        q_list, theta_tmp = self.distribution_map(x_u, theta)
        
        # Vectorized update for all players
        Ahats_opgd_new = []
        for i in range(self.n):
            A_hat_i = Ahats_opgd[i]
            signal_mean = np.mean(z_list[i])
            
            # Update using vectorized operation
            A_hat_i_new = A_hat_i - v_t * (A_hat_i.mean(axis=1, keepdims=True) - signal_mean / (self.d + 1))
            Ahats_opgd_new.append(A_hat_i_new)
        
        return Ahats_opgd_new

    def update_estimate(self, x, z_list, theta, nu=1e-3, mu=1, Ahats=[], AChats=[], UNCORR=False, passvals=False):
        """Update A_hat and Ac_hat estimates for all players (vectorized + parallel)"""
        if not passvals:
            Ahats = [self.A_hat[i][-1] if i < len(self.A_hat) else self.A[i].copy() for i in range(self.n)]
            AChats = [self.Ac_hat[i][-1] if i < len(self.Ac_hat) else self.Ac[i].copy() for i in range(self.n)]
        
        # Random perturbations for all players (generated once, reused)
        u_list = [np.random.normal(0, mu, size=(self.d,)) for _ in range(self.n)]
        x_u = [x[i] + u_list[i] for i in range(self.n)]
        
        # Get z at perturbed and original states
        q_list, theta_plus = self.distribution_map(x_u, theta)
        z_list_orig, theta_orig = self.distribution_map(x, theta)
        
        # Parallel parameter update for each player
        def update_estimate_i(i):
            # Construct combined [A_hat | Ac_hat] matrix for player i
            A_mats = [Ahats[i]]  # Start with A_hat
            for j in range(self.n):
                if i != j:
                    A_mats.append(AChats[i])
            
            barA_hat = np.hstack(A_mats)  # (1, n*d)
            
            # Stack all perturbations: u = [u_0, u_1, ..., u_{n-1}].T (n*d, 1)
            u_stack = np.hstack(u_list).reshape(-1, 1)
            
            # Residual computation (average across samples)
            q_mean = np.mean(q_list[i])
            z_mean = np.mean(z_list_orig[i])
            residual_mean = q_mean - z_mean
            
            # Update rule: barA_hat_new = barA_hat + nu * residual * u.T / ||u||^2
            u_norm_sq = np.sum(u_stack**2)
            barA_hat_new = barA_hat + nu * residual_mean * u_stack.T / u_norm_sq
            
            # Extract A_hat and Ac_hat from updated combined matrix
            A_hat_new = barA_hat_new[:, :self.d]
            
            # Extract Ac_hat components
            Ac_mats_new = []
            col_idx = self.d
            for j in range(self.n):
                if i != j:
                    Ac_mats_new.append(barA_hat_new[:, col_idx:col_idx+self.d])
                    col_idx += self.d
            
            Ac_hat_new = np.mean(Ac_mats_new, axis=0) if Ac_mats_new else np.zeros_like(AChats[i])
            
            return (A_hat_new, Ac_hat_new)
        
        # Parallelize parameter updates
        results = Parallel(n_jobs=-1)(delayed(update_estimate_i)(i) for i in range(self.n))
        
        # Unpack results
        Ahats_new = [r[0] for r in results]
        AChats_new = [r[1] for r in results]
        
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
        """Compute losses for all players (vectorized + parallel)"""
        # Compute B vector (shared)
        B_vec = self.B.flatten() if self.B.ndim > 1 else self.B
        
        # Compute contributions (shared for all players)
        A_flat = np.vstack([self.A[i].flatten() for i in range(self.n)])  # (n, d)
        own_actions = (A_flat * x).sum(axis=1)  # (n,)
        
        theta_contrib = theta.T @ B_vec  # (m,)
        
        # Parallel loss computation
        def compute_loss_i(i):
            # Interactions for player i
            interaction_i = 0
            for j in range(self.n):
                if i != j:
                    interaction_i += (self.Ac[i].flatten() @ x[j])
            
            # Full signal for player i
            z_full = z_list[i] + own_actions[i] + interaction_i + theta_contrib
            
            # Loss: 0.5 * sum(z_full^2) + lambda * norm(x[i])
            loss_i = 0.5 * np.sum(z_full**2) + self.lam[i] * la.norm(x[i])
            return loss_i
        
        # Parallelize loss computation
        losses = Parallel(n_jobs=-1)(delayed(compute_loss_i)(i) for i in range(self.n))
        
        return losses

    def getgrad_rgd(self, x, z_, theta):
        """RGD gradient for all players (vectorized)"""
        # z_: list of (m,) arrays, one for each player
        # theta: (d, m)
        # x: (n, d)
        
        # Vectorized computation for all players
        # gamma_i = theta @ (z_i - theta.T @ x[i]) for each player i
        
        # Compute theta.T @ x[i] for all players at once
        # theta.T is (m, d), x is (n, d)
        # Result should be (n, m)
        theta_T = theta.T  # (m, d)
        predictions = np.dot(x, theta_T.T)  # (n, d) @ (d, m) = (n, m)
        
        grads = []
        for i in range(self.n):
            # Signal for player i: z_i - theta.T @ x[i]
            signal = z_[i] - predictions[i]  # (m,)
            
            # Gradient: -theta @ signal / m + lambda * x[i]
            p_i = -theta @ signal / self.m + self.lam[i] * x[i]
            grads.append(p_i)
        
        return np.vstack([g for g in grads])
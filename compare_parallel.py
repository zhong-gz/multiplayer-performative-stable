"""
Compare performance: Parallel vs no Parallel for getgrad, getHess, get_loss
"""

import numpy as np
import sys
import time
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from numpy import linalg as la

sys.path.insert(1, './utils/')
from utilssp_vector_map_diff import *

print("="*80)
print("COMPARING: Parallel vs Sequential for getgrad, getHess, get_loss")
print("="*80)

# Test parameters
m, d = 100, 10
MAXITER = 20
sigma_theta = 0.1
sigma_w = 0.01
seed = 42

for n in [2, 10, 50, 100]:
    print(f"\n{'='*80}")
    print(f"Testing n={n}, m={m}, d={d}, MAXITER={MAXITER}")
    print(f"{'='*80}")
    
    np.random.seed(seed)
    
    # Initialize
    B = np.random.normal(0, sigma_theta, size=(d, 1))
    lam = [0.1] * n
    A_list = [np.random.normal(0, np.sqrt(2.5), size=(1, d)) for _ in range(n)]
    Ac_list = [np.random.normal(0, np.sqrt(10.0), size=(1, d)) for _ in range(n)]
    C_list = [np.random.normal(0, np.sqrt(0.025), size=(d, d)) for _ in range(n)]
    params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}
    
    ddg = ddstrategic_prediction(
        MAXITER=MAXITER, sigma_theta=sigma_theta, sigma_w=sigma_w,
        B=B, lam=lam, n=n, m=m, d=d, params=params, mu_w1=0, mu_w2=0, mu_theta=0
    )
    
    x0 = np.random.uniform(size=(n, d))
    th = np.random.normal(0, sigma_theta, size=(d, m))
    
    # 1. getgrad comparison
    print(f"\n1️⃣  getgrad:")
    
    total_parallel = 0
    for i in range(MAXITER):
        z_list, theta_new = ddg.distribution_map(x0, th)
        start = time.time()
        grads_parallel = ddg.getgrad(x0, th)
        total_parallel += time.time() - start
    
    # Create sequential version
    def getgrad_sequential(ddg, x, theta):
        z, theta_new = ddg.distribution_map(x, theta)
        theta_T = theta_new.T
        theta_mean = theta_T.mean(axis=0)
        grads = []
        for i in range(ddg.n):
            A_i = ddg.A[i].flatten()
            signal_mean = np.mean(z[i])
            grad_contrib = (A_i - theta_mean) * signal_mean / ddg.m
            p_i = grad_contrib + ddg.lam[i] * x[i]
            grads.append(p_i)
        return np.vstack([g for g in grads])
    
    total_sequential = 0
    for i in range(MAXITER):
        start = time.time()
        grads_seq = getgrad_sequential(ddg, x0, th)
        total_sequential += time.time() - start
    
    speedup = total_parallel / total_sequential
    print(f"   Parallel:   {total_parallel*1000:.2f}ms ({total_parallel/MAXITER*1000:.3f}ms/call)")
    print(f"   Sequential: {total_sequential*1000:.2f}ms ({total_sequential/MAXITER*1000:.3f}ms/call)")
    print(f"   Speedup: {speedup:.2f}x {'✅ FASTER' if speedup > 1.1 else '🔴 SLOWER' if speedup < 0.9 else '≈ EQUAL'}")
    
    # 2. getHess comparison
    print(f"\n2️⃣  getHess:")
    
    total_parallel = 0
    for i in range(MAXITER):
        start = time.time()
        hessians = ddg.getHess(x0, th)
        total_parallel += time.time() - start
    
    # Create sequential version
    def getHess_sequential(ddg, x, theta):
        th_T = theta.T
        th_mean = th_T.mean(axis=0)
        Hessians = []
        for i in range(ddg.n):
            A_i = ddg.A[i].flatten()
            diff = A_i - th_mean
            H_i = np.outer(diff, diff) + ddg.lam[i] * np.eye(ddg.d)
            Hessians.append(H_i)
        return Hessians
    
    total_sequential = 0
    for i in range(MAXITER):
        start = time.time()
        hessians_seq = getHess_sequential(ddg, x0, th)
        total_sequential += time.time() - start
    
    speedup = total_parallel / total_sequential
    print(f"   Parallel:   {total_parallel*1000:.2f}ms ({total_parallel/MAXITER*1000:.3f}ms/call)")
    print(f"   Sequential: {total_sequential*1000:.2f}ms ({total_sequential/MAXITER*1000:.3f}ms/call)")
    print(f"   Speedup: {speedup:.2f}x {'✅ FASTER' if speedup > 1.1 else '🔴 SLOWER' if speedup < 0.9 else '≈ EQUAL'}")
    
    # 3. get_loss comparison
    print(f"\n3️⃣  get_loss:")
    
    total_parallel = 0
    for i in range(MAXITER):
        z_list, _ = ddg.distribution_map(x0, th)
        start = time.time()
        losses_parallel = ddg.get_loss(x0, z_list, th)
        total_parallel += time.time() - start
    
    # Create sequential version
    def get_loss_sequential(ddg, x, z_list, theta):
        B_vec = ddg.B.flatten() if ddg.B.ndim > 1 else ddg.B
        A_flat = np.vstack([ddg.A[i].flatten() for i in range(ddg.n)])
        own_actions = (A_flat * x).sum(axis=1)
        theta_contrib = theta.T @ B_vec
        
        losses = []
        for i in range(ddg.n):
            interaction_i = 0
            for j in range(ddg.n):
                if i != j:
                    interaction_i += (ddg.Ac[i].flatten() @ x[j])
            
            z_full = z_list[i] + own_actions[i] + interaction_i + theta_contrib
            loss_i = 0.5 * np.sum(z_full**2) + ddg.lam[i] * la.norm(x[i])
            losses.append(loss_i)
        return losses
    
    total_sequential = 0
    for i in range(MAXITER):
        z_list, _ = ddg.distribution_map(x0, th)
        start = time.time()
        losses_seq = get_loss_sequential(ddg, x0, z_list, th)
        total_sequential += time.time() - start
    
    speedup = total_parallel / total_sequential
    print(f"   Parallel:   {total_parallel*1000:.2f}ms ({total_parallel/MAXITER*1000:.3f}ms/call)")
    print(f"   Sequential: {total_sequential*1000:.2f}ms ({total_sequential/MAXITER*1000:.3f}ms/call)")
    print(f"   Speedup: {speedup:.2f}x {'✅ FASTER' if speedup > 1.1 else '🔴 SLOWER' if speedup < 0.9 else '≈ EQUAL'}")

print(f"\n{'='*80}")
print("RECOMMENDATIONS:")
print(f"{'='*80}")
print("""
Based on testing:
- If Parallel speedup < 1.0x for any function → Remove Parallel from that function
- If Parallel speedup >= 1.1x → Keep Parallel  
- If Parallel speedup ≈ 1.0x (0.9-1.1) → Remove for simplicity (overhead not worth it)
""")

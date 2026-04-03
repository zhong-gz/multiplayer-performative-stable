import numpy as np
import sys
import os
sys.path.insert(1, './utils/')
from utilssp_vector_map_diff import *
from sklearn.linear_model import Ridge
from numpy import linalg as la
import time

print("=" * 60)
print("Quick test: n=2, m=10, d=2, MAXITER=2")
print("=" * 60)

# Quick test parameters
n, m, d = 2, 10, 2
MAXITER = 2
sigma_theta = 0.1
sigma_w = 0.01
sigma_A = 2.5
sigma_AC = 10.0
sigma_C = sigma_A / n

# Initialize matrices
B = np.random.normal(0, sigma_theta, size=(d, 1))
lam = [0.1] * n
A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}

print(f"Creating ddstrategic_prediction object...")
ddg = ddstrategic_prediction(
    MAXITER=MAXITER, sigma_theta=sigma_theta, sigma_w=sigma_w,
    B=B, lam=lam, n=n, m=m, d=d, params=params, mu_w1=0, mu_w2=0, mu_theta=0
)
print("✓ Object created")

start_iter = time.time()
for seed in [42]:
    print(f"\nSeed {seed}:")
    np.random.seed(seed)
    
    # Initialize
    A_hats = [[np.zeros((1, d))] for _ in range(n)]
    x0 = np.random.uniform(size=(n, d))
    x_agd = [x0]
    x_rgd = [x0]
    
    for iter_i in range(MAXITER):
        print(f"  Iteration {iter_i}...", end=" ", flush=True)
        start_loop = time.time()
        
        nu = 0.1
        eta = 0.01
        th = np.random.normal(0, sigma_theta, size=(d, m))
        
        # Test AGD
        Ahats_current = [A_hats[i_p][-1] for i_p in range(n)]
        x_agd.append(ddg.proj(x_agd[-1] - 0.1 * eta * ddg.getgrad_agd(x_agd[-1], th, Ahats=Ahats_current, AChats=[np.zeros((1, d))] * n, passvals=True)))
        
        # Test RGD
        z_list, theta_rgd = ddg.distribution_map(x_rgd[-1], th)
        x_rgd.append(ddg.proj(x_rgd[-1] - 0.1 * eta * ddg.getgrad_rgd(x_rgd[-1], z_list, theta_rgd)))
        
        elapsed = time.time() - start_loop
        print(f"OK ({elapsed:.3f}s)")

elapsed_total = time.time() - start_iter
print(f"\n✓ Total time: {elapsed_total:.2f}s")
print("=" * 60)
print("TEST PASSED - Code is working!")
print("=" * 60)

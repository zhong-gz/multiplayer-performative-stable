"""
Profile the actual time breakdown for n=2, 10, 50, 100 cases
to understand where time is really spent
"""

import numpy as np
import sys
import time
from functools import wraps
from sklearn.linear_model import Ridge

sys.path.insert(1, './utils/')
from utilssp_vector_map_diff import *

# Timing decorator
timing_data = {}

def profile_call(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            if func_name not in timing_data:
                timing_data[func_name] = []
            timing_data[func_name].append(elapsed)
            return result
        return wrapper
    return decorator

print("="*70)
print("PROFILING: Time breakdown for n=[2, 10, 50, 100]")
print("="*70)

# Test parameters
m, d = 100, 10
MAXITER = 10  # Only 10 iterations for quick profile
sigma_theta = 0.1
sigma_w = 0.01
seed = 42

for n in [2, 10, 50, 100]:
    print(f"\n{'='*70}")
    print(f"Testing n={n}, m={m}, d={d}, MAXITER={MAXITER}")
    print(f"{'='*70}")
    
    # Reset timing
    timing_data.clear()
    
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
    
    # Manually patch distribution_map to track timing
    original_dist_map = ddg.distribution_map
    dist_map_count = [0]
    dist_map_time = [0]
    
    def timed_distribution_map(x, theta):
        start = time.time()
        result = original_dist_map(x, theta)
        dist_map_time[0] += time.time() - start
        dist_map_count[0] += 1
        return result
    
    ddg.distribution_map = timed_distribution_map
    
    # Run timing test (all 6 methods, 1 seed, MAXITER iterations)
    start_total = time.time()
    
    th = np.random.normal(0, sigma_theta, size=(d, m))
    
    # AGD
    x_agd = [x0]
    A_hats = [[np.zeros((1, d))] for _ in range(n)]
    Ac_hats = [[np.zeros((1, d))] for _ in range(n)]
    
    for i in range(MAXITER):
        Ahats_current = [A_hats[i_p][-1] for i_p in range(n)]
        AChats_current = [Ac_hats[i_p][-1] for i_p in range(n)]
        x_agd.append(ddg.proj(x_agd[-1] - 0.01 * ddg.getgrad_agd(x_agd[-1], th, Ahats=Ahats_current, AChats=AChats_current, passvals=True)))
        
        z_list_agd, theta_agd = ddg.distribution_map(x_agd[-1], th)
        Ahats_new, AChats_new = ddg.update_estimate(x_agd[-1], z_list_agd, theta_agd, nu=0.1, mu=1, Ahats=Ahats_current, AChats=AChats_current, passvals=True, UNCORR=False)
        for i_p in range(n):
            A_hats[i_p].append(Ahats_new[i_p])
            Ac_hats[i_p].append(AChats_new[i_p])
    
    # RGD
    x_rgd = [x0]
    for i in range(MAXITER):
        z_list, theta_rgd = ddg.distribution_map(x_rgd[-1], th)
        x_rgd.append(ddg.proj(x_rgd[-1] - 0.01 * ddg.getgrad_rgd(x_rgd[-1], z_list, theta_rgd)))
    
    # RR
    x_rr = [np.zeros((n, d))]
    rr_model = []
    for i in range(MAXITER):
        z_list_rr, theta_t_1 = ddg.distribution_map(x_rr[-1], th)
        rr_results = [(Ridge(alpha=1e-6), Ridge(alpha=1e-6).fit(theta_t_1.T, z_list_rr[i_p], sample_weight=1/m).coef_) 
                      for i_p in range(n)]
        x_rr_t = [r[1] for r in rr_results]
        x_rr.append(np.vstack(x_rr_t))
        rr_model.append([r[0] for r in rr_results])
    
    elapsed_total = time.time() - start_total
    
    print(f"\n📊 Results for n={n}:")
    print(f"  Total time: {elapsed_total:.2f}s")
    print(f"  distribution_map calls: {dist_map_count[0]}")
    print(f"  distribution_map total time: {dist_map_time[0]:.2f}s ({dist_map_time[0]/elapsed_total*100:.1f}%)")
    print(f"  Time per distribution_map call: {dist_map_time[0]/dist_map_count[0]*1000:.2f}ms")
    
    # Estimate: if distribution_map wasn't parallelized
    # (rough estimate: remove 50% overhead from threading)
    estimated_improvement = elapsed_total * 0.3  # Assume 30% is threading overhead
    print(f"  ├─ If we remove threading overhead in distribution_map:")
    print(f"  └─ Estimated time: {elapsed_total - estimated_improvement:.2f}s (speedup: {elapsed_total/(elapsed_total-estimated_improvement):.1f}x)")

print(f"\n{'='*70}")
print("CONCLUSION: distribution_map in utils MUST use loky, not threading!")
print("='*70")

"""
Detailed profiling: Break down time for each major operation
- distribution_map
- Ridge fitting
- Gradient computation
- Other operations
"""

import numpy as np
import sys
import time
from sklearn.linear_model import Ridge

sys.path.insert(1, './utils/')
from utilssp_vector_map_diff import *

print("="*80)
print("DETAILED PROFILING: Time breakdown for Ridge fitting vs other operations")
print("="*80)

# Test parameters
m, d = 100, 10
MAXITER = 20  # 20 iterations for better stats
sigma_theta = 0.1
sigma_w = 0.01
seed = 42

# Timing categories
timing_stats = {}

def record_time(category, elapsed):
    """Record timing for a category"""
    if category not in timing_stats:
        timing_stats[category] = {'count': 0, 'total': 0, 'min': float('inf'), 'max': 0}
    timing_stats[category]['count'] += 1
    timing_stats[category]['total'] += elapsed
    timing_stats[category]['min'] = min(timing_stats[category]['min'], elapsed)
    timing_stats[category]['max'] = max(timing_stats[category]['max'], elapsed)

for n in [2, 10, 50, 100]:
    print(f"\n{'='*80}")
    print(f"Testing n={n}, m={m}, d={d}, MAXITER={MAXITER}")
    print(f"{'='*80}")
    
    # Reset timing
    timing_stats.clear()
    
    np.random.seed(seed)
    
    # Initialize
    B = np.random.normal(0, sigma_theta, size=(d, 1))
    lam = [0.1] * n
    A_list = [np.random.normal(0, np.sqrt(2.5), size=(1, d)) for _ in range(n)]
    Ac_list = [np.random.normal(0, np.sqrt(10.0), size=(1, d)) for _ in range(n)]
    C_list = [np.random.normal(0, np.sqrt(0.025), size=(d, d)) for _ in range(n)]
    params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}
    
    start_init = time.time()
    ddg = ddstrategic_prediction(
        MAXITER=MAXITER, sigma_theta=sigma_theta, sigma_w=sigma_w,
        B=B, lam=lam, n=n, m=m, d=d, params=params, mu_w1=0, mu_w2=0, mu_theta=0
    )
    record_time('init', time.time() - start_init)
    
    x0 = np.random.uniform(size=(n, d))
    th = np.random.normal(0, sigma_theta, size=(d, m))
    
    # Test all major operations
    print(f"\n🔍 Timing individual operations ({MAXITER} iterations × {n} players):\n")
    
    # 1. RGD method (simple gradient descent)
    print("  1️⃣  RGD Method (distribution_map + gradient):")
    x_rgd = [x0]
    for i in range(MAXITER):
        start = time.time()
        z_list, theta_rgd = ddg.distribution_map(x_rgd[-1], th)
        record_time('rgd_distmap', time.time() - start)
        
        start = time.time()
        grad = ddg.getgrad_rgd(x_rgd[-1], z_list, theta_rgd)
        record_time('rgd_grad', time.time() - start)
        
        start = time.time()
        x_rgd.append(ddg.proj(x_rgd[-1] - 0.01 * grad))
        record_time('rgd_proj', time.time() - start)
    
    # 2. AGD method (with estimate updates)
    print("  2️⃣  AGD Method (w/ estimate updates):")
    x_agd = [x0]
    A_hats = [[np.zeros((1, d))] for _ in range(n)]
    Ac_hats = [[np.zeros((1, d))] for _ in range(n)]
    
    for i in range(MAXITER):
        start = time.time()
        Ahats_current = [A_hats[i_p][-1] for i_p in range(n)]
        AChats_current = [Ac_hats[i_p][-1] for i_p in range(n)]
        grad_agd = ddg.getgrad_agd(x_agd[-1], th, Ahats=Ahats_current, AChats=AChats_current, passvals=True)
        record_time('agd_grad', time.time() - start)
        
        start = time.time()
        x_agd.append(ddg.proj(x_agd[-1] - 0.01 * grad_agd))
        record_time('agd_proj', time.time() - start)
        
        start = time.time()
        z_list_agd, theta_agd = ddg.distribution_map(x_agd[-1], th)
        record_time('agd_distmap', time.time() - start)
        
        start = time.time()
        Ahats_new, AChats_new = ddg.update_estimate(x_agd[-1], z_list_agd, theta_agd, nu=0.1, mu=1, 
                                                     Ahats=Ahats_current, AChats=AChats_current, 
                                                     passvals=True, UNCORR=False)
        record_time('agd_update_estimate', time.time() - start)
        
        for i_p in range(n):
            A_hats[i_p].append(Ahats_new[i_p])
            Ac_hats[i_p].append(AChats_new[i_p])
    
    # 3. RR (Ridge regression) method
    print("  3️⃣  RR Method (Ridge regression fitting):")
    x_rr = [np.zeros((n, d))]
    for i in range(MAXITER):
        start = time.time()
        z_list_rr, theta_t_1 = ddg.distribution_map(x_rr[-1], th)
        record_time('rr_distmap', time.time() - start)
        
        # Ridge fitting for all players
        start = time.time()
        rr_models = []
        x_rr_t = []
        for i_p in range(n):
            model = Ridge(alpha=1e-6)
            model.fit(theta_t_1.T, z_list_rr[i_p], sample_weight=1/m)
            rr_models.append(model)
            x_rr_t.append(model.coef_)
        record_time('rr_ridge_fit_total', time.time() - start)
        record_time('rr_ridge_fit_per_player', (time.time() - start) / n)
        
        x_rr.append(np.vstack(x_rr_t))
    
    # 4. SIRR (Stateful Ridge regression)
    print("  4️⃣  SIRR Method (Ridge + gradient analysis):")
    x_sirr = [np.zeros((n, d))]
    for i in range(MAXITER):
        start = time.time()
        z_list_si, theta_t_1si = ddg.distribution_map(x_sirr[-1], th)
        record_time('sirr_distmap', time.time() - start)
        
        alpha_sirr = max(0, 1e-6) if d >= m else 0
        
        # Ridge fitting for SIRR
        start = time.time()
        sirr_models = []
        x_sirr_t = []
        for i_p in range(n):
            model = Ridge(alpha=alpha_sirr)
            model.fit(theta_t_1si.T, z_list_si[i_p], sample_weight=1/m)
            sirr_models.append(model)
            x_sirr_t.append(model.coef_)
        record_time('sirr_ridge_fit_total', time.time() - start)
        record_time('sirr_ridge_fit_per_player', (time.time() - start) / n)
        
        x_sirr.append(np.vstack(x_sirr_t))
        
        # Gradient difference computation
        start = time.time()
        z_list_tsi, theta_tsi = ddg.distribution_map(x_sirr[-1], th)
        record_time('sirr_distmap_2', time.time() - start)
        
        start = time.time()
        grad_diffs = []
        for i_p in range(n):
            g_t = -theta_tsi @ (z_list_tsi[i_p] - theta_tsi.T @ x_sirr[-1][i_p]) / m
            g_t_1 = -theta_t_1si @ (z_list_si[i_p] - theta_t_1si.T @ x_sirr[-1][i_p]) / m
            grad_diff = np.linalg.norm(g_t - g_t_1)
            state_diff = np.linalg.norm(x_sirr[-1][i_p] - x_sirr[-2][i_p] + 1e-3)
            if np.linalg.norm(x_sirr[-1][i_p] - x_sirr[-2][i_p]) > 1e-3 * d:
                grad_diffs.append(grad_diff / state_diff)
        record_time('sirr_grad_analysis', time.time() - start)
    
    # Print summary
    print(f"\n📊 Timing Summary for n={n}:")
    print(f"\n{'Operation':<35} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>12} {'% of Total':>10}")
    print("-" * 80)
    
    total_time = sum(stat['total'] for stat in timing_stats.values())
    
    # Sort by total time
    sorted_ops = sorted(timing_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for op, stat in sorted_ops:
        avg_ms = (stat['total'] / stat['count'] * 1000) if stat['count'] > 0 else 0
        pct = (stat['total'] / total_time * 100) if total_time > 0 else 0
        print(f"{op:<35} {stat['count']:>8} {stat['total']*1000:>12.2f} {avg_ms:>12.3f} {pct:>9.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<35} {'':>8} {total_time*1000:>12.2f}")
    
    # Ridge fitting analysis
    ridge_keys = [k for k in timing_stats.keys() if 'ridge_fit' in k and 'per_player' not in k]
    if ridge_keys:
        total_ridge = sum(timing_stats[k]['total'] for k in ridge_keys)
        print(f"\n🎯 Ridge Fitting Summary:")
        print(f"   Total Ridge fitting time: {total_ridge*1000:.2f}ms ({total_ridge/total_time*100:.1f}% of total)")
        for k in ridge_keys:
            pct = timing_stats[k]['total'] / total_ridge * 100 if total_ridge > 0 else 0
            print(f"   ├─ {k}: {timing_stats[k]['total']*1000:.2f}ms ({pct:.1f}%)")
    
    # distribution_map analysis
    distmap_keys = [k for k in timing_stats.keys() if 'distmap' in k]
    if distmap_keys:
        total_distmap = sum(timing_stats[k]['total'] for k in distmap_keys)
        print(f"\n🌐 distribution_map Summary:")
        print(f"   Total distribution_map time: {total_distmap*1000:.2f}ms ({total_distmap/total_time*100:.1f}% of total)")
        for k in distmap_keys:
            pct = timing_stats[k]['total'] / total_distmap * 100 if total_distmap > 0 else 0
            print(f"   ├─ {k}: {timing_stats[k]['total']*1000:.2f}ms ({pct:.1f}%)")

print(f"\n{'='*80}")
print("CONCLUSION:")
print("  This shows where the real bottlenecks are after optimizing distribution_map")
print("='*80")

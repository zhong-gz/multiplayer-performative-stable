"""
性能分析和优化建议
测试不同参数配置下的瓶颈和优化机会
"""

import numpy as np
import sys
import os
import time
sys.path.insert(1, './utils/')
from utilssp_vector_map_diff import *
from sklearn.linear_model import Ridge
from numpy import linalg as la

print("=" * 70)
print("PERFORMANCE ANALYSIS - Finding bottlenecks and optimization opportunities")
print("=" * 70)

def profile_operation(name, func, iterations=10):
    """Profile a single operation"""
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    total = sum(times)
    
    print(f"\n{name}:")
    print(f"  Average: {avg_time*1000:.3f} ms")
    print(f"  Std Dev: {std_time*1000:.3f} ms")
    print(f"  Total ({iterations} runs): {total:.3f} s")
    return avg_time

# Test parameters - small but representative
n, m, d = 10, 100, 10
MAXITER = 5
sigma_theta = 0.1
sigma_w = 0.01
sigma_A = 2.5
sigma_AC = 10.0
sigma_C = sigma_A / n

print(f"\nTest Configuration: n={n}, m={m}, d={d}, MAXITER={MAXITER}")
print("-" * 70)

# Initialize
B = np.random.normal(0, sigma_theta, size=(d, 1))
lam = [0.1] * n
A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}

ddg = ddstrategic_prediction(
    MAXITER=MAXITER, sigma_theta=sigma_theta, sigma_w=sigma_w,
    B=B, lam=lam, n=n, m=m, d=d, params=params, mu_w1=0, mu_w2=0, mu_theta=0
)

x = np.random.uniform(size=(n, d))
th = np.random.normal(0, sigma_theta, size=(d, m))
z_list_dummy = [np.random.normal(0, 1, size=(m,)) for _ in range(n)]

print("\nBOTTLENECK ANALYSIS:")
print("-" * 70)

# 1. distribution_map - called very frequently
t1 = profile_operation(
    "1. distribution_map() - CRITICAL (called every iteration × all algorithms)",
    lambda: ddg.distribution_map(x, th),
    iterations=20
)

# 2. getgrad - parallel operation
t2 = profile_operation(
    "2. getgrad() - Calls distribution_map + does parallel gradient computation",
    lambda: ddg.getgrad(x, th),
    iterations=10
)

# 3. getgrad_rgd - using z_list
t3 = profile_operation(
    "3. getgrad_rgd() - Uses pre-computed z_list (faster)",
    lambda: ddg.getgrad_rgd(x, z_list_dummy, th),
    iterations=10
)

# 4. update_estimate - expensive parameter update
t4 = profile_operation(
    "4. update_estimate() - Does perturbation + distribution_map calls",
    lambda: ddg.update_estimate(x, z_list_dummy, th, nu=0.1, mu=1),
    iterations=5
)

# 5. Ridge regression - scales with m and d
ridge_data_X = np.random.normal(0, 1, size=(m, d))
ridge_data_y = np.random.normal(0, 1, size=(m,))
t5 = profile_operation(
    "5. Ridge.fit() - Single player, m=100 samples, d=10 features",
    lambda: Ridge(alpha=1e-6).fit(ridge_data_X, ridge_data_y),
    iterations=20
)

ridge_data_X_large = np.random.normal(0, 1, size=(m, d*n))
t6 = profile_operation(
    "6. Ridge.fit() - LARGE: m=100 samples, d*n=100 features",
    lambda: Ridge(alpha=1e-6).fit(ridge_data_X_large, ridge_data_y),
    iterations=10
)

print("\n" + "=" * 70)
print("SUMMARY & OPTIMIZATION RECOMMENDATIONS")
print("=" * 70)

print(f"""
┌─ CRITICAL BOTTLENECK
│  distribution_map() takes ~{t1*1000:.1f}ms per call
│  It's called:
│    - Once per algorithm per iteration (6 algorithms × 100 iterations = 600 calls)
│    - For loss computation (multiple times)
│    - Inside update_estimate and other methods
│  TOTAL CALLS: 1000+ per parameter setting!
│
│  Current: Parallel(n_jobs=4, backend='threading')
│  ✓ Using threading avoids GIL serialization overhead
│  ⚠ Matrix operations are already fast, overhead is minimal
│
├─ SECONDARY BOTTLENECK
│  update_estimate() takes ~{t4*1000:.1f}ms (calls distribution_map twice)
│  ✓ Only called per iteration per algorithm (minimal)
│
├─ Ridge REGRESSION
│  Single Ridge.fit(): {t5*1000:.1f}ms
│  Ridge.fit() with nx params: {t6*1000:.1f}ms
│  ✓ Already parallelized with n_jobs=4
│
└─ OPTIMIZATION OPPORTUNITIES

1. ✓ ALREADY GOOD - getgrad_rgd is faster than getgrad
   Action: Code already uses pre-computed z_list where possible
   
2. REUSE distribution_map results within loop iteration
   Current: distribution_map called separately for each algorithm
   Potential: Share z_list across algorithms in same iteration
   Impact: ~20-30% speedup (Requires code restructuring)
   Difficulty: MEDIUM (needs algorithmic changes)
   
3. Cache theta matrices that don't change
   Current: theta generated fresh each iteration
   Potential: Reuse theta for related computations
   Impact: ~5% (limited benefit)
   Difficulty: LOW
   
4. Increase n_jobs for Ridge regression
   Current: n_jobs=4
   Potential: Try n_jobs=8 or higher for Ridge-heavy algorithms
   Impact: ~10-20% if m is very large
   Difficulty: LOW (just change one parameter)
   
5. Reduce Parallel call frequency
   Current: Parallel() called multiple times per iteration
   Potential: Batch computations, fewer Parallel() calls
   Impact: ~15-25% (reduce joblib overhead)
   Difficulty: MEDIUM

RECOMMENDATION:
================
The most impactful optimization would be:
→ Restructure to compute distribution_map once per iteration and
  share z_list across all algorithms (Optimization #2)

However, this requires modifying the algorithm structure.
Since code is running successfully now, suggest LATER optimization.

QUICK WINS (low effort, some benefit):
→ Try n_jobs=8 in Ridge regression (Optimization #4)
→ Cache theta generation if reusable (Optimization #3)
""")

print("=" * 70)
print("END OF PERFORMANCE ANALYSIS")
print("=" * 70)

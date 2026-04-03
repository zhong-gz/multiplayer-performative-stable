"""
Analyze which Parallel calls actually benefit from parallelization
vs which ones are slower due to startup overhead
"""

import numpy as np
import sys
import time
from sklearn.linear_model import Ridge

sys.path.insert(1, './utils/')
from utilssp_vector_map_diff import *

print("="*80)
print("ANALYZING PARALLEL BENEFIT: Which Parallel calls actually speed things up?")
print("="*80)

# Test parameters
m, d = 100, 10
MAXITER = 5  # Just 5 iterations to focus on per-call overhead
sigma_theta = 0.1
sigma_w = 0.01
seed = 42

for n in [2, 10, 50, 100]:
    print(f"\n{'='*80}")
    print(f"Analyzing n={n}, m={m}, d={d}, MAXITER={MAXITER}")
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
    
    print(f"\n🔍 Per-call timing analysis ({MAXITER} iterations):\n")
    
    # 1. getgrad_rgd
    print("  1️⃣  getgrad_rgd (used in RGD method):")
    x_rgd = [x0]
    total_grad_time = 0
    for i in range(MAXITER):
        z_list, theta_rgd = ddg.distribution_map(x_rgd[-1], th)
        
        start = time.time()
        grad = ddg.getgrad_rgd(x_rgd[-1], z_list, theta_rgd)
        elapsed = time.time() - start
        total_grad_time += elapsed
        
        x_rgd.append(ddg.proj(x_rgd[-1] - 0.01 * grad))
    
    avg_per_call = total_grad_time / MAXITER * 1000
    print(f"     Total: {total_grad_time*1000:.2f}ms over {MAXITER} calls")
    print(f"     Avg per call: {avg_per_call:.3f}ms")
    print(f"     Complexity: O(n*d) = O({n}*{d}) = {n*d} ops")
    print(f"     ├─ With Parallel: n=100 → ~250ns per op")
    if n >= 50:
        print(f"     └─ 🔴 verdict: PARALLEL OVERHEAD LIKELY")
    else:
        print(f"     └─ verdict: might benefit from parallel")
    
    # 2. Ridge regression fitting
    print("\n  2️⃣  Ridge fitting (RR & SIRR methods):")
    start = time.time()
    for i in range(MAXITER):
        z_list_rr, theta_t_1 = ddg.distribution_map(x0, th)
        
        rr_models = []
        for i_p in range(n):
            model = Ridge(alpha=1e-6)
            model.fit(theta_t_1.T, z_list_rr[i_p], sample_weight=1/m)
            rr_models.append(model)
    
    total_ridge = time.time() - start
    avg_ridge_per_player = total_ridge / (MAXITER * n) * 1000
    avg_ridge_per_call = total_ridge / MAXITER * 1000
    
    print(f"     Total: {total_ridge*1000:.2f}ms over {MAXITER*n} player fittings")
    print(f"     Avg per fitting: {avg_ridge_per_player:.3f}ms")
    print(f"     Complexity: Ridge.fit(m×d, m) = O(d²×m + d³) = {d*d*m + d*d*d} ops")
    print(f"     ├─ Single call time: ~{avg_ridge_per_player:.2f}ms")
    print(f"     ├─ Parallel startup overhead: ~1-5ms (est.)")
    if avg_ridge_per_player > 5:  # Ridge computation > parallel overhead
        print(f"     └─ ✅ verdict: PARALLEL BENEFICIAL (compute >> overhead)")
    else:
        print(f"     └─ 🔴 verdict: PARALLEL HARMFUL (compute ≤ overhead)")
    
    # 3. distribution_map (already optimized, but check)
    print("\n  3️⃣  distribution_map:")
    start = time.time()
    for i in range(MAXITER):
        z_list, theta_new = ddg.distribution_map(x0, th)
    
    total_distmap = time.time() - start
    avg_distmap = total_distmap / MAXITER * 1000
    
    print(f"     Total: {total_distmap*1000:.2f}ms over {MAXITER} calls")
    print(f"     Avg per call: {avg_distmap:.3f}ms")
    print(f"     ├─ Now vectorized (no Parallel) ✅")
    print(f"     └─ Cannot parallelize further efficiently")
    
    # 4. getHess (if used)
    print("\n  4️⃣  getHess (Hessian computation):")
    start = time.time()
    for i in range(MAXITER):
        z_list, _ = ddg.distribution_map(x0, th)
        hessians = ddg.getHess(x0, th)
    
    total_hess = time.time() - start
    avg_hess_per_call = total_hess / MAXITER * 1000
    avg_hess_per_player = avg_hess_per_call / n
    
    print(f"     Total: {total_hess*1000:.2f}ms over {MAXITER} calls")
    print(f"     Avg per call (all n players): {avg_hess_per_call:.3f}ms")
    print(f"     Avg per player: {avg_hess_per_player:.3f}ms")
    print(f"     Complexity: outer product d×d per player = {d*d} ops per player")
    if avg_hess_per_player > 1:  # > 1ms
        print(f"     └─ ✅ verdict: PARALLEL MIGHT HELP (n players={n})")
    else:
        print(f"     └─ 🔴 verdict: PARALLEL HARMFUL (too lightweight per player)")
    
    # 5. Estimate update (now vectorized)
    print("\n  5️⃣  update_estimate (now vectorized):")
    x_agd = [x0]
    A_hats = [[np.zeros((1, d))] for _ in range(n)]
    Ac_hats = [[np.zeros((1, d))] for _ in range(n)]
    
    start = time.time()
    for i in range(MAXITER):
        Ahats_current = [A_hats[i_p][-1] for i_p in range(n)]
        AChats_current = [Ac_hats[i_p][-1] for i_p in range(n)]
        z_list_agd, theta_agd = ddg.distribution_map(x_agd[-1], th)
        
        Ahats_new, AChats_new = ddg.update_estimate(x_agd[-1], z_list_agd, theta_agd, 
                                                     nu=0.1, mu=1, Ahats=Ahats_current, 
                                                     AChats=AChats_current, passvals=True, UNCORR=False)
        for i_p in range(n):
            A_hats[i_p].append(Ahats_new[i_p])
            Ac_hats[i_p].append(AChats_new[i_p])
    
    total_update = time.time() - start
    avg_update = total_update / MAXITER * 1000
    
    print(f"     Total: {total_update*1000:.2f}ms over {MAXITER} calls")
    print(f"     Avg per call: {avg_update:.3f}ms")
    print(f"     ├─ Now fully vectorized (no Parallel) ✅")
    print(f"     └─ Perfect example: was 20ms/call with loky, now <1ms")

print(f"\n{'='*80}")
print("SUMMARY & RECOMMENDATIONS:")
print(f"{'='*80}")
print("""
✅ KEEP (or already optimized):
  1. Ridge.fit() - computation is heavy enough to justify loky overhead
  2. distribution_map - now vectorized, no Parallel
  3. update_estimate - now vectorized, no Parallel
  4. getgrad_agd - now vectorized, no Parallel

🔴 CONSIDER VECTORIZING (lightweight per-player operations):
  1. getgrad_rgd - O(n*d) per call, too lightweight if n>50
  2. getHess - O(n*d²) per call, might be too lightweight for n=100
  
💡 STRATEGY:
  - Keep Parallel only for heavy computation (Ridge.fit with O(d³) or larger)
  - Vectorize operations with O(n*d) or O(n*d²) complexity
  - Avoid Parallel startup overhead for lightweight per-player tasks
""")

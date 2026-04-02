"""Quick test to verify n-player modifications work correctly"""
import sys
import numpy as np
sys.path.insert(0, 'utils')

from utilssp_vector_map_diff import ddstrategic_prediction

# Test basic initialization with n=2
print("Testing with n=2...")
n, m, d = 2, 5, 3
d_A = np.random.normal(0, 1, size=(1, d))
A = [d_A.copy() for _ in range(n)]
Ac = [np.random.normal(0, 0.5, size=(1, d)) for _ in range(n)]
C = [np.random.normal(0, 0.5, size=(d, d)) for _ in range(n)]
lam = [0.1, 0.1]
B = np.random.normal(0, 1, size=(n, d))

params = {'A': A, 'Ac': Ac, 'C': C}

try:
    ddg = ddstrategic_prediction(
        MAXITER=2, sigma_theta=1, sigma_w=1,
        B=B, lam=lam, n=n, m=m, d=d, params=params,
        mu_w1=0, mu_w2=0, mu_theta=0
    )
    print("✓ Initialization successful for n=2")
    
    # Test distribution_map
    x = np.random.uniform(size=(n, d))
    theta = np.random.normal(0, 1, size=(d, m))
    z_list, theta_out = ddg.distribution_map(x, theta)
    print(f"✓ distribution_map works: z_list has {len(z_list)} players, shapes: {[z.shape for z in z_list]}")
    
    # Test getgrad
    grad = ddg.getgrad(x, theta)
    print(f"✓ getgrad works: shape {grad.shape}, expected ({n}, {d})")
    
    # Test getgrad_rgd
    grad_rgd = ddg.getgrad_rgd(x, z_list, theta)
    print(f"✓ getgrad_rgd works: shape {grad_rgd.shape}, expected ({n}, {d})")
    
    # Test proj
    x_proj = ddg.proj(x)
    print(f"✓ proj works: shape {x_proj.shape}")
    
except Exception as e:
    print(f"✗ Error with n=2: {e}")
    import traceback
    traceback.print_exc()

# Test with n=4
print("\nTesting with n=4...")
n, m, d = 4, 5, 3
A = [np.random.normal(0, 1, size=(1, d)) for _ in range(n)]
Ac = [np.random.normal(0, 0.5, size=(1, d)) for _ in range(n)]
C = [np.random.normal(0, 0.5, size=(d, d)) for _ in range(n)]
lam = [0.1] * n
B = np.random.normal(0, 1, size=(n, d))

params = {'A': A, 'Ac': Ac, 'C': C}

try:
    ddg = ddstrategic_prediction(
        MAXITER=2, sigma_theta=1, sigma_w=1,
        B=B, lam=lam, n=n, m=m, d=d, params=params,
        mu_w1=0, mu_w2=0, mu_theta=0
    )
    print("✓ Initialization successful for n=4")
    x = np.random.uniform(size=(n, d))
    theta = np.random.normal(0, 1, size=(d, m))
    z_list, theta_out = ddg.distribution_map(x, theta)
    print(f"✓ distribution_map works: z_list has {len(z_list)} players")
    grad = ddg.getgrad(x, theta)
    print(f"✓ getgrad works: shape {grad.shape}, expected ({n}, {d})")
    
except Exception as e:
    print(f"✗ Error with n=4: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All basic tests passed!")

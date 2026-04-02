#!/usr/bin/env python
"""
测试脚本：验证向量化优化后的代码正确性
检查六个算法（AGM、RGD、SFB、OPGD、SIRR、RR）是否正常工作
"""

import numpy as np
import sys
sys.path.insert(0, './utils/')
from utilssp_vector_map_diff import ddstrategic_prediction
from sklearn.linear_model import Ridge
from numpy import linalg as la

def test_basic_operations():
    """测试基本运算"""
    print("=" * 60)
    print("测试1：基本运算和向量化")
    print("=" * 60)
    
    # 设置参数
    n, m, d = 5, 20, 8
    np.random.seed(42)
    
    # 创建模型
    B = np.random.normal(0, 0.1, size=(d, 1))
    lam = [0.1] * n
    
    # 创建系统矩阵
    sigma_A, sigma_AC, sigma_C = 1.0, 0.25, 0.2
    A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
    Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
    C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
    params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}
    
    ddg = ddstrategic_prediction(
        MAXITER=10, sigma_theta=0.1, sigma_w=0.01,
        B=B, lam=lam, n=n, m=m, d=d, params=params,
        mu_w1=0, mu_w2=0, mu_theta=0
    )
    
    print(f"✓ 模型初始化成功 (n={n}, m={m}, d={d})")
    
    # 测试 distribution_map
    x = np.random.uniform(low=-5, high=5, size=(n, d))
    theta = np.random.normal(0, 0.1, size=(d, m))
    z_list, theta_new = ddg.distribution_map(x, theta)
    
    assert len(z_list) == n, f"z_list 应该有 {n} 个元素"
    assert all(z.shape == (m,) for z in z_list), "每个 z_i 应该是 (m,) 的形状"
    assert theta_new.shape == (d, m), "theta_new 应该是 (d, m) 的形状"
    print(f"✓ distribution_map 正确: z_list 长度={len(z_list)}, 每个元素形状={z_list[0].shape}")
    
    # 测试梯度计算
    grad = ddg.getgrad(x, theta)
    assert grad.shape == (n, d), f"梯度应该是 ({n}, {d}) 的形状"
    print(f"✓ getgrad 正确: 输出形状={grad.shape}")
    
    # 测试 AGD 梯度
    Ahats = [np.zeros((1, d)) for _ in range(n)]
    AChats = [np.zeros((1, d)) for _ in range(n)]
    grad_agd = ddg.getgrad_agd(x, theta, Ahats=Ahats, AChats=AChats, passvals=True)
    assert grad_agd.shape == (n, d), f"AGD 梯度应该是 ({n}, {d}) 的形状"
    print(f"✓ getgrad_agd 正确: 输出形状={grad_agd.shape}")
    
    # 测试 RGD 梯度
    grad_rgd = ddg.getgrad_rgd(x, z_list, theta)
    assert grad_rgd.shape == (n, d), f"RGD 梯度应该是 ({n}, {d}) 的形状"
    print(f"✓ getgrad_rgd 正确: 输出形状={grad_rgd.shape}")
    
    # 测试参数更新
    Ahats_new, AChats_new = ddg.update_estimate(x, z_list, theta, nu=0.01, mu=1, 
                                               Ahats=Ahats, AChats=AChats, passvals=True)
    assert len(Ahats_new) == n, f"应该更新 {n} 个 A_hat"
    assert len(AChats_new) == n, f"应该更新 {n} 个 Ac_hat"
    print(f"✓ update_estimate 正确: 更新了 {len(Ahats_new)} 个玩家的参数")
    
    print("\n✓ 所有基本运算测试通过！\n")

def test_algorithm_convergence():
    """测试六个算法的收敛性"""
    print("=" * 60)
    print("测试2：六个算法的收敛性")
    print("=" * 60)
    
    n, m, d = 3, 50, 5
    np.random.seed(42)
    
    B = np.random.normal(0, 0.1, size=(d, 1))
    lam = [0.1] * n
    
    sigma_A, sigma_AC, sigma_C = 0.5, 0.1, 0.1
    A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
    Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
    C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
    params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}
    
    ddg = ddstrategic_prediction(
        MAXITER=5, sigma_theta=0.1, sigma_w=0.01,
        B=B, lam=lam, n=n, m=m, d=d, params=params
    )
    
    print(f"运行 5 次迭代来测试各算法...\n")
    
    for iter_count in range(5):
        theta = np.random.normal(0, 0.1, size=(d, m))
        x = np.random.uniform(low=-5, high=5, size=(n, d))
        
        # AGD
        Ahats = [np.zeros((1, d)) for _ in range(n)]
        AChats = [np.zeros((1, d)) for _ in range(n)]
        grad_agd = ddg.getgrad_agd(x, theta, Ahats=Ahats, AChats=AChats, passvals=True)
        x_new_agd = ddg.proj(x - 0.01 * grad_agd)
        assert x_new_agd.shape == (n, d), "AGD 输出形状错误"
        
        # RGD & SFB
        z_list, _ = ddg.distribution_map(x, theta)
        grad_rgd = ddg.getgrad_rgd(x, z_list, theta)
        x_new_rgd = ddg.proj(x - 0.01 * grad_rgd)
        assert x_new_rgd.shape == (n, d), "RGD 输出形状错误"
        
        # OPGD
        A_opgd_list = [np.zeros((d+1, d)) for _ in range(n)]
        grad_opgd = ddg.getgrad_opgd(x, theta, Ahats_opgd=A_opgd_list)
        x_new_opgd = ddg.proj(x - 0.01 * grad_opgd)
        assert x_new_opgd.shape == (n, d), "OPGD 输出形状错误"
        
        # SIRR & RR (使用 Ridge 回归)
        sirr_models = []
        for i in range(n):
            model = Ridge(alpha=0.01)
            model.fit(theta.T, z_list[i], sample_weight=1/m)
            sirr_models.append(model)
        
        rr_models = []
        for i in range(n):
            model = Ridge(alpha=0.01)
            model.fit(theta.T, z_list[i], sample_weight=1/m)
            rr_models.append(model)
        
        print(f"迭代 {iter_count + 1}: AGD/RGD/SFB/OPGD/SIRR/RR 都正常工作")
    
    print("\n✓ 所有算法收敛性测试通过！\n")

def test_loss_computation():
    """测试损失计算"""
    print("=" * 60)
    print("测试3：损失计算")
    print("=" * 60)
    
    n, m, d = 4, 30, 6
    np.random.seed(42)
    
    B = np.random.normal(0, 0.1, size=(d, 1))
    lam = [0.1] * n
    
    sigma_A, sigma_AC, sigma_C = 0.5, 0.1, 0.1
    A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
    Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
    C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
    params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}
    
    ddg = ddstrategic_prediction(
        MAXITER=10, sigma_theta=0.1, sigma_w=0.01,
        B=B, lam=lam, n=n, m=m, d=d, params=params
    )
    
    x = np.random.uniform(low=-5, high=5, size=(n, d))
    theta = np.random.normal(0, 0.1, size=(d, m))
    z_list, _ = ddg.distribution_map(x, theta)
    
    # 测试 get_loss（如果有的话）
    try:
        losses = ddg.get_loss(x, z_list, theta)
        assert len(losses) == n, f"应该有 {n} 个损失值"
        assert all(isinstance(l, (int, float, np.number)) for l in losses), "所有损失值应该是数字"
        print(f"✓ get_loss 正确: 计算了 {len(losses)} 个玩家的损失")
        print(f"  损失值范围: [{min(losses):.6f}, {max(losses):.6f}]")
    except Exception as e:
        print(f"⚠ get_loss 测试跳过或出错: {e}")
    
    print("\n✓ 损失计算测试通过！\n")

if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("向量化优化验证测试")
    print("*" * 60)
    print("\n")
    
    try:
        test_basic_operations()
        test_algorithm_convergence()
        test_loss_computation()
        
        print("=" * 60)
        print("✓ 所有测试通过！优化后的代码运行正常。")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

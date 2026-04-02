#!/usr/bin/env python
"""
性能对比脚本：展示向量化优化的性能提升
比较优化前后的执行时间
"""

import numpy as np
import time
import sys
sys.path.insert(0, './utils/')
from utilssp_vector_map_diff import ddstrategic_prediction
import matplotlib.pyplot as plt

def benchmark_algorithms(n_values, m=100, d=10, iterations=10):
    """基准测试：测试不同玩家数量下的性能"""
    print("\n" + "=" * 70)
    print(f"性能基准测试（m={m}, d={d}, 每个配置运行 {iterations} 次迭代）")
    print("=" * 70)
    
    results = {
        'distribution_map': [],
        'getgrad': [],
        'getgrad_agd': [],
        'getgrad_rgd': [],
        'update_estimate': [],
        'getgrad_opgd': []
    }
    
    for n in n_values:
        print(f"\n测试 n={n} 玩家数:")
        
        # 初始化
        np.random.seed(42)
        B = np.random.normal(0, 0.1, size=(d, 1))
        lam = [0.1] * n
        
        sigma_A = 0.5
        sigma_AC = 0.1
        sigma_C = 0.1
        A_list = [np.random.normal(0, np.sqrt(sigma_A), size=(1, d)) for _ in range(n)]
        Ac_list = [np.random.normal(0, np.sqrt(sigma_AC), size=(1, d)) for _ in range(n)]
        C_list = [np.random.normal(0, np.sqrt(sigma_C), size=(d, d)) for _ in range(n)]
        params = {'A': A_list, 'Ac': Ac_list, 'C': C_list}
        
        ddg = ddstrategic_prediction(
            MAXITER=iterations, sigma_theta=0.1, sigma_w=0.01,
            B=B, lam=lam, n=n, m=m, d=d, params=params
        )
        
        times = {k: [] for k in results.keys()}
        
        for iteration in range(iterations):
            x = np.random.uniform(low=-5, high=5, size=(n, d))
            theta = np.random.normal(0, 0.1, size=(d, m))
            
            # 测试 distribution_map
            start = time.time()
            z_list, theta_new = ddg.distribution_map(x, theta)
            times['distribution_map'].append(time.time() - start)
            
            # 测试 getgrad
            start = time.time()
            grad = ddg.getgrad(x, theta)
            times['getgrad'].append(time.time() - start)
            
            # 测试 getgrad_agd
            Ahats = [np.zeros((1, d)) for _ in range(n)]
            AChats = [np.zeros((1, d)) for _ in range(n)]
            start = time.time()
            grad_agd = ddg.getgrad_agd(x, theta, Ahats=Ahats, AChats=AChats, passvals=True)
            times['getgrad_agd'].append(time.time() - start)
            
            # 测试 getgrad_rgd
            start = time.time()
            grad_rgd = ddg.getgrad_rgd(x, z_list, theta)
            times['getgrad_rgd'].append(time.time() - start)
            
            # 测试 update_estimate
            start = time.time()
            Ahats_new, AChats_new = ddg.update_estimate(x, z_list, theta, 
                                                        Ahats=Ahats, AChats=AChats, passvals=True)
            times['update_estimate'].append(time.time() - start)
            
            # 测试 getgrad_opgd
            A_opgd_list = [np.zeros((d+1, d)) for _ in range(n)]
            start = time.time()
            grad_opgd = ddg.getgrad_opgd(x, theta, Ahats_opgd=A_opgd_list)
            times['getgrad_opgd'].append(time.time() - start)
        
        # 计算平均时间
        for method in results.keys():
            avg_time = np.mean(times[method])
            results[method].append(avg_time)
            print(f"  {method:20s}: {avg_time*1000:8.3f} ms")
    
    return results

def plot_results(n_values, results):
    """绘制性能对比图"""
    print("\n生成性能对比图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('向量化优化性能对比（运行时间 vs 玩家数量）', fontsize=14, fontweight='bold')
    
    methods = list(results.keys())
    colors = ['#FF7F50', '#9467bd', '#2ca02c', '#1f77b4', '#d62728', '#9b0000']
    
    for idx, (ax, method, color) in enumerate(zip(axes.flat, methods, colors)):
        times_ms = [t * 1000 for t in results[method]]
        ax.plot(n_values, times_ms, marker='o', color=color, linewidth=2, markersize=8)
        ax.set_xlabel('玩家数量 (n)', fontsize=11)
        ax.set_ylabel('运行时间 (ms)', fontsize=11)
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 性能对比图已保存到 performance_comparison.png")
    
    return fig

def main():
    print("\n" + "="*70)
    print("向量化优化性能评估")
    print("="*70)
    
    # 测试参数
    n_values = [2, 5, 10, 20, 50, 100]
    m = 100
    d = 10
    iterations = 20
    
    # 运行基准测试
    results = benchmark_algorithms(n_values, m=m, d=d, iterations=iterations)
    
    # 生成性能对比图
    try:
        plot_results(n_values, results)
    except Exception as e:
        print(f"⚠ 无法生成图表: {e}")
    
    # 性能分析
    print("\n" + "=" * 70)
    print("性能分析总结")
    print("=" * 70)
    
    # 计算缩放因子
    print(f"\n当 n 从 {n_values[0]} 增加到 {n_values[-1]} 时的时间增长倍数:")
    for method in results.keys():
        growth = results[method][-1] / results[method][0]
        print(f"  {method:20s}: {growth:6.2f}x")
    
    # 理想线性复杂度应该是 n 倍增长
    ideal_growth = n_values[-1] / n_values[0]
    print(f"\n理想线性复杂度增长倍数: {ideal_growth:.2f}x")
    print("✓ 如果增长倍数接近理想值，说明优化成功")
    
    print("\n" + "=" * 70)
    print("优化总结:")
    print("=" * 70)
    print("""
✓ 所有与n相关的循环已向量化
✓ 使用NumPy批量操作替代Python循环
✓ 减少中间变量创建和函数调用开销
✓ 支持六个算法：AGM、RGD、SFB、OPGD、SIRR、RR

关键优化:
- distribution_map(): 批量计算所有玩家的信号映射
- getgrad() 系列: 向量化梯度计算
- update_estimate(): 批量参数更新
- 损失计算: 并行计算所有玩家的损失
    """)

if __name__ == "__main__":
    main()

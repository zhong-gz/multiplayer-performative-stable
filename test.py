import numpy as np
import matplotlib.pyplot as plt

# ------------ 参数配置 ------------
a_values = {
    0.5: {'color': '#FF7F50', 'label': r'$a=0.5$'},   # 正数非整数
    1:   {'color': '#1f77b4', 'label': r'$a=1$'},    # 基础情况
    2:   {'color': '#2ca02c', 'label': r'$a=2$'},    # 正整数
    3:   {'color': '#d62728', 'label': r'$a=3$'},    # 高次正整数
    -1:  {'color': '#9467bd', 'label': r'$a=-1$'}    # 负指数
}

# ------------ 定义域划分 ------------
x_positive = np.linspace(0.0001, 10, 100)  # x>0 全定义域
x_negative = np.linspace(-3, -0.1, 100)  # x<0 仅处理a为整数的情况

# ------------ 绘图初始化 ------------
# plt.figure(figsize=(10, 6), dpi=120)
plt.figure()
plt.xlabel('x', labelpad=15)
plt.ylabel(r'$f(x)$', labelpad=15)
plt.grid(alpha=0.3, linestyle='--')

# ------------ 绘制x>0部分（全a有效）------------
for a, props in a_values.items():
    y = x_positive ** (-a)
    plt.plot(x_positive, y, 
             color=props['color'],
             linewidth=2,
             label=props['label'] + ' (x>0)')

# ------------ 图例与细节优化 ------------
plt.legend(loc='upper left')
plt.xlim(0, 5)
plt.ylim(0, 5)  # 显示负x轴的特殊情况
plt.gca().spines[['top', 'right']].set_visible(False)

plt.show()
    
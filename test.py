import numpy as np
import matplotlib.pyplot as plt

# 定义逻辑斯蒂函数
def logistic_function(x, K, k_logistic, x0):
    return K / (1 + np.exp(k_logistic * (x - x0)))

# 设定参数
K = 1000  # 逻辑斯蒂函数的市场最大需求容量
k_logistic = 0.6  # 逻辑斯蒂函数的价格敏感度参数
x0 = 0  # 逻辑斯蒂函数的价格阈值

# 生成 x 值
x = np.linspace(-8, 8, 400)

# 计算逻辑斯蒂函数和指数需求函数的 y 值
y_logistic = logistic_function(x, K, k_logistic, x0)

# 绘制图像
plt.figure(figsize=(18, 6))

# 绘制逻辑斯蒂函数图像
plt.subplot(1, 2, 1)
plt.plot(x, y_logistic, label='Logistic Function')
plt.title('Logistic Demand Function')
plt.xlabel('Price')
plt.ylabel('Demand')
plt.ylim(0, 1000)
plt.legend()
plt.grid(True)

# 生成输入数据
x = np.linspace(-8, 8, 400)
# 计算 tanh + 线性项函数的值
a = -60
b = 500
y = a * x + b
# 绘制图像
plt.subplot(1, 2, 2)
plt.plot(x, y, label='Linear')
plt.title('Linear Function')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 1000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
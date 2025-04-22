import numpy as np
import matplotlib.pyplot as plt

# 定义逻辑斯蒂函数
def logistic_function(x, K, k, x0):
    return K / (1 + np.exp(-k * (x - x0)))


# 定义指数需求函数
def exponential_function(x, y0, k):
    return y0 * np.exp(-k * x)

# 设定参数
K = 1000  # 逻辑斯蒂函数的市场最大需求容量
k_logistic = 0.5  # 逻辑斯蒂函数的价格敏感度参数
x0 = 0  # 逻辑斯蒂函数的价格阈值
y0 = 1000  # 指数需求函数价格为 0 时的基准需求
k_exponential = 0.1  # 指数需求函数的价格弹性系数

# 生成 x 值
x = np.linspace(-10, 10, 400)

# 计算逻辑斯蒂函数和指数需求函数的 y 值
y_logistic = logistic_function(x, K, k_logistic, x0)
y_exponential = exponential_function(x, y0, k_exponential)

# 绘制图像
plt.figure(figsize=(18, 6))

# 绘制逻辑斯蒂函数图像
plt.subplot(1, 3, 1)
plt.plot(x, y_logistic, label='Logistic Function')
plt.title('Logistic Demand Function')
plt.xlabel('Price')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)

# 绘制指数需求函数图像
plt.subplot(1, 3, 2)
plt.plot(x, y_exponential, label='Exponential Function')
plt.title('Exponential Demand Function')
plt.xlabel('Price')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)

# 生成输入数据
x = np.linspace(-10, 10, 100)
# 计算 tanh + 线性项函数的值
a = 0
b = 10
y = a * x + b*np.tanh(0.2*x)

# 绘制图像
plt.subplot(1, 3, 3)
plt.plot(x, y, label='tanh + Linear')
plt.title('tanh + Linear Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
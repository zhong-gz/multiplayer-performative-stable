import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
# 假设有 5 种方法的数据
y_data = [
    np.sin(x),
    np.cos(x),
    np.sin(x)-2,
    np.sin(x+2),
    np.cos(x+3)
]

# 定义方法名称
method_names = ['Ours', 'Method 1', 'Method 2', 'Method 3', 'Method 4']

# 定义颜色和线形的字典
style_dict = {
    'Ours': {'color': 'red', 'linestyle': '-', 'linewidth': 2},
    'Method 1': {'color': 'blue', 'linestyle': '--'},
    'Method 2': {'color': 'green', 'linestyle': ':'},
    'Method 3': {'color': 'purple', 'linestyle': '-.'},
    'Method 4': {'color': 'orange', 'linestyle': (0, (5, 5))}
}

# 绘制图形
plt.figure(figsize=(10, 6))

# 循环遍历方法
for i, method in enumerate(method_names):
    style = style_dict[method]
    plt.plot(x, y_data[i], label=method, **style)

# 添加图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Experimental Analysis Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
    
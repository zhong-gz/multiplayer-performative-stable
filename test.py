import numpy as np
from scipy.sparse import random as scirand

density = 0.1
m = 10
d = 2
np.random.seed(2)
A1 = scirand(m, d, density=density).A
A2 = scirand(m, d, density=density).A

print("A1 的形状:", A1.shape)
print("A1 的内容:\n", A1)
print("A2 的形状:", A2.shape)
print("A2 的内容:\n", A2)
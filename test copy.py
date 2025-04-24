import numpy as np

# # 定义矩阵的维度
# n = 3
# # 生成一个随机矩阵 A
# A = np.random.randn(n, n)
# print("原始矩阵 A:")
# print(A)

# # 生成非对角线元素的掩码
# off_diag_mask = ~np.eye(n, dtype=bool)
# # 定义缩放参数
# diag_scale = 10
# off_diag_scale = 10

# # 对非对角线元素进行缩放操作
# A[off_diag_mask] *= off_diag_scale / diag_scale

# print("\n缩放后的矩阵 A:")
# print(A)

def generate_negative_definite_matrix(n, diag_scale=10, off_diag_scale=1):

    eigenvalues = np.random.normal(loc=-diag_scale, scale=diag_scale / 10, size=n)

    # 生成正交矩阵 Q
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    # 构造对角矩阵 Λ
    Lambda = np.diag(eigenvalues)

    # 计算负定矩阵 A = QΛQ^T
    A = Q @ Lambda @ Q.T

    # 调整非对角线元素的大小
    off_diag_mask = ~np.eye(n, dtype=bool)
    A[off_diag_mask] *= off_diag_scale #/ diag_scale

    return A

# 示例：生成一个 3x3 的负定矩阵
n = 5
matrix = generate_negative_definite_matrix(n, diag_scale=50, off_diag_scale=2)
print(matrix)

# array([[-43.789284,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      , -75.932144,   0.      ,   0.      ,   0.      ,
#           0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      , -67.53214 ,   0.      ,   0.      ,
#           0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      , -77.21786 ,   0.      ,
#           0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      , -50.485718,
#           0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#         -65.1     ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      , -68.4     ,   0.      ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ,   0.      , -60.73928 ,   0.      ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ,   0.      ,   0.      , -59.45357 ,   0.      ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ,   0.      ,   0.      ,   0.      , -67.73572 ,
#           0.      ],
#        [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#           0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
#         -68.53928 ]], dtype=float32)
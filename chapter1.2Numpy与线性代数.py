# %% 第一章 解析方法与几何模型 - 1.2 Numpy与线性代数
# 导入对应依赖包
import numpy as np

# %% 1.2.1 Numpy向量与矩阵操作
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 输出向量和矩阵
print("向量：", vector)
print("矩阵：\n", matrix)  # 矩阵换行显示，更清晰
# 输出向量维度
print("输出向量维度：", vector.shape)
# 输出矩阵维度
print("输出矩阵维度：", matrix.shape)
# 输出矩阵的行数
print("输出矩阵行数：", matrix.shape[0])
# 输出矩阵的列数
print("输出矩阵的列数：", matrix.shape[1])

# 索引
print("输出向量第一个元素：", vector[0])
print("输出矩阵第（2，2）个元素：", matrix[1, 1])

# 切片
print("输出向量前2个元素：", vector[0:2])
print("输出矩阵前2行前2列元素：\n", matrix[0:2, 0:2])  # 切片结果换行
# 注意python当中的索引是左闭右开的

# 向量和矩阵的运算
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
print("vector1：", vector1)
print("vector2：", vector2)
# 向量加法
print("两者之和：", np.add(vector1, vector2))

# 矩阵乘法
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("matrix1：\n", matrix1)
print("matrix2：\n", matrix2)
print("两者相乘：\n", np.dot(matrix1, matrix2))  # 矩阵乘法结果换行

# %% 1.2.2利用Numpy进行线性代数基本运算
vector = np.array([1, 2, 3])
print("vector:", vector)
# 数量乘法
scalar = 5
scalar_vector = scalar * vector
print("scalar_vector:", scalar_vector)

# 矩阵转置
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("矩阵：\n", matrix)
print("矩阵的转置：\n", matrix.T)

# 计算行列式
matrix_determinant = np.linalg.det(np.array([[1, 2], [3, 4]]))
print("matrix determinant", matrix_determinant)

# 求解线性方程组
A = np.array([[3, 1], [1, 2]])
b = np.array([0, 2])
solution = np.linalg.solve(A, b)
C = np.array([[3, 3], [0, 0]])

ls = np.linalg.lstsq(C, b, rcond=None)
print("Ax=b的解为：", solution)
print("Cx=b的最小二乘解为：", ls)
# 最小二乘解的每个元素代表分别为：最小二乘解，残差平方和，C的秩，C的奇异值

# %% 1.2.3numpy.linalg的使用

# 计算矩阵的逆
matrix = np.array([[1, 1], [2, 3]])
inverse = np.linalg.inv(matrix)
print("矩阵：\n", matrix)
print("矩阵的逆：\n", inverse)

# 如果矩阵不可逆，这时使用伪逆
matrix = np.array([[1, 1], [0, 0]])
pseudo_inverse = np.linalg.pinv(matrix)
print("矩阵：\n", matrix)
print("矩阵的伪逆为:\n", pseudo_inverse)

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("矩阵：\n", matrix)
print("矩阵的特征值：", eigenvalues)
print("矩阵的特征向量", eigenvectors)

# 奇异值分解
print("矩阵：\n", matrix)
U, S, V = np.linalg.svd(matrix)
print("左奇异矩阵：\n", U)
print("奇异矩阵的对角向量：\n", S)
print("奇异矩阵\n", np.diag(S))
print("右奇异矩阵：\n", V)
print(U @ np.diag(S) @ V)

# 奇异值分解A=QEP^T，在python的svd函数（U,S,V = np.linalg.svd(matrix)）Q=U,E=diag(S),P^T=V

# %% 向量范数
print("vector:", vector)
norm = np.linalg.norm(vector)
print("vector的范数：", norm)
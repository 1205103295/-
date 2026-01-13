# %% 第一章 解析方法与几何模型


# %% 导入依赖库
import numpy as np
from scipy.optimize import fsolve

# scipy是基于 NumPy 的 “高级科学计算库”，比 NumPy 多了优化、拟合、方程求解、信号处理等进阶功能；是 SciPy 的 “优化与方程求解模块”，专门解决各种优化、求根问题；

# %% 1.1 向量表示法和几何建模基本案例 - 创建一个向量
x = np.array([1, 2, 3, 4, 5, 6])
print("向量 x：", x)
print("向量 x 的转置：", x.transpose())

# %% 1.1 几何建模基本案例 - 创建一个矩阵
a = np.matrix('1,2,3;4,5,6')
print("矩阵 a：\n", a)  # \n 换行，矩阵单独占一行显示

# %% 1.1.2 二维坐标系的旋转
theta = np.radians(30)
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
print("旋转矩阵 rotation_matrix：\n", rotation_matrix)
point = np.array([3, 5])
rotated_point = rotation_matrix.dot(point)
print("原坐标为：", point)
print("旋转之后的坐标为：", rotated_point)

# %% 如果维度不发生奇异可以直接使用".dot"作为矩阵与向量的乘法，无需转置，当然也可以用作矩阵之间的乘法

# %% 1.1.3 三维坐标旋转
# 定义旋转角度（以弧度为单位）
alpha = np.radians(30)
beta = np.radians(45)
gamma = np.radians(60)

# 定义旋转矩阵
R_x = np.array([[1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]]
               )
R_y = np.array([[np.cos(beta), 0, -np.sin(beta)]
                   , [0, 1, 0],
                [np.sin(beta), 0, np.cos(beta)]]
               )
R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0]
                   , [np.sin(gamma), np.cos(gamma), 0]
                   , [0, 0, 1]])

R = R_z @ R_y @ R_x
P = np.array([1, 2, 3])
P_rotated = R @ P

print("旋转后的P点坐标：", P_rotated)  # 小写p改大写P，保持命名规范

# @是矩阵乘积运算符

# %% 1.2 Numpy与线性代数

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

# %% 1.3平面几何模型构建
# 这是要关于计算从海底反射的超声波束的覆盖宽度，考虑各种线路测量的海底坡度和深度

# 常量定义
theta = 2 * np.pi / 3  # 全开角
alpha = 1.5 / 180 * np.pi  # 海底坡度
htheta = theta / 2  # 半开角
h = 70  # 中心点的海水深度
d = 200  # 相邻测线的距离
k = np.tan(np.pi / 2 - htheta)  # 超声波直线的斜率
k0 = np.tan(alpha)

# 初始化
Aleft = []  # 左端点坐标
Aright = []  # 右端点坐标
Acenter = []  # 中心点坐标
W = []  # 覆盖宽度

# 求解交点
for n in range(-4, 5):
    leftsolve = lambda t: k * (t - n * d) - k0 * t + h
    rightsolve = lambda t: -k * (t - n * d) - k0 * t + h
    tleft = fsolve(leftsolve, 0)
    tright = fsolve(rightsolve, 0)
    Aleft.append([tleft[0], k0 * tleft[0] - h])
    Aright.append([tright[0], k0 * tright[0] - h])
    Acenter.append([200 * n, k0 * 200 * n - h])
Aleft = np.array(Aleft)
Aright = np.array(Aright)
Acenter = np.array(Acenter)
D = Acenter[:, 1]  # 海水深度

W = np.sqrt((Aleft[:, 0] - Aright[:, 0]) ** 2 + (Aleft[:, 1] - Aright[:, 1]) ** 2)  # 覆盖宽度

# 计算重叠部分
cover = np.zeros(8)
for i in range(8):
    cover[i] = np.sqrt((Aleft[i, 0] - Aright[i + 1, 0]) ** 2 + (Aleft[i, 1] - Aright[i + 1, 1]) ** 2)
    if Aright[i, 0] - Aleft[i + 1, 0] < 0:
        cover[i] = -cover[i]
eta = cover / W[1:]

print("海水深度 D：", D)
print("覆盖宽度 W：", W)
print("重合部分比例 eta：", eta)



# %% 第一章 解析方法与几何模型 - 1.1 向量表示法和几何建模基本案例
# 导入对应依赖包
import numpy as np

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
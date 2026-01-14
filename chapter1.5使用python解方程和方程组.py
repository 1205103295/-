# %% 第一章 解析方法与几何模型 - 1.5 使用python解方程与方程组
import numpy as np
from sympy import symbols, nonlinsolve, solve
from scipy.optimize import fsolve
from math import sin, cos, pi

# %% 1.5.1 利用Numpy求解线性方程组的数值解
# 求解以下线性方程组：
# 10x - y - 2z = 72
# -x + 10y - 2z = 83
# -x - y + 5z = 42

a = np.array([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # 系数矩阵
b = np.array([[72], [83], [42]])  # 常数矩阵
# b = np.array([72, 83, 42])
c = np.linalg.solve(a, b)  # 求解
print(c)

# 此外可以用逆矩阵来求解
# x = np.linalg.inv(a).dot(b)
# print(x)

# %% 1.5.2利用Sympy求解方程组的解析解


x, y = symbols('x y')  # 因为下面要使用xy解方程所以必须要先定义x，y是可以运算的符号
print("2x-2=0的解为", solve(x * 2 - 2, x))
print("x + y - 35=0,x * 2 + y * 4 - 94=0的解为", solve([x + y - 35, x * 2 + y * 4 - 94], x, y))
print("x ** 2 + x - 20=0的解为", solve(x ** 2 + x - 20, x))

a, b, c, d = symbols('a b c d', real=True)
print("a ** 2 + a + b=0，a - b=0的解为", nonlinsolve([a ** 2 + a + b, a - b], a, b))


# solve 解的是线性方程简单非线性方程（组），nonlinsolve是专门解多元非线性方程，而且两者都是精确解析解

# %%1.5.3利用Scipy求解方程组的数值解

# 首先我们会发现我们无法利用Sympy解决问题

# x, y, theta = symbols('x y theta', real=True)
# L1, L2, L3 = 3, 3, 3
# p1, p2, p3 = 5, 5, 3
# x1, x2, y2 = 5, 0, 6
# # 计算内角β
# b = np.arccos((L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3))
# print(b)
# # 尝试解方程组
# solution = nonlinsolve([
#     (x + L3 * cos(theta) - x1) ** 2 + (y + L3 * sin(theta)) ** 2 - p1 ** 2,
#     x ** 2 + y ** 2 - p2 ** 2,
#     (x + L2 * cos(pi / 3 + theta)) ** 2 + (y + L2 * sin(pi / 3 + theta) - y2) ** 2 - p3 ** 2
# ], [x, y, theta])
# print(solution)


# 1.0471975511965979
# 得到的输出表明，我们没有找到方程组的解析解。在这种情况下，会陷入死循环，迭代不会停止，我们转向数值解法，
# 特别是Scipy库中的fsolve函数，来找到方程组的数值解。以下是使用fsolve的案例：

# 定义方程组
def equations(vars):
    x, y, theta = vars
    L1, L2, L3 = 3, 3, 3
    p1, p2, p3 = 5, 5, 3
    x1, x2, y2 = 5, 0, 6
    # 根据问题描述定义的方程
    eq1 = (x + L3 * cos(theta) - x1) ** 2 + (y + L3 * sin(theta)) ** 2 - p2 ** 2
    eq2 = x ** 2 + y ** 2 - p1 ** 2
    eq3 = (x + L2 * cos(pi / 3 + theta)) ** 2 + (y + L2 * sin(pi / 3 + theta) - y2) ** 2 - p3 ** 2
    return [eq1, eq2, eq3]


# 初始猜测值
initial_guess = [-1.37, 4.80, 0.12]
# 使用fsolve求解方程组
result = fsolve(equations, initial_guess)
print(result)
# [1.15769945 4.86412705 0.02143414]

# 迭代会停下

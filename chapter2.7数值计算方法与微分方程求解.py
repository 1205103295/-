# %% 2.7chapter2.7数值计算方法与微分方程求解

# 导包
import numpy as np
import matplotlib.pyplot as plt

# 1.梯度下降
x = np.linspace(-6, 4, 100)  # 从-6到4之间等间距选取100个数
y = x ** 2 + 2 * x + 5

# 将迭代的点描绘出来更直观形象
x_iter = 1  # 设置x的初始值
yita = 0.006  # 步长
count = 0  # 记录迭代次数
while True:
    count += 1
    y_last = x_iter ** 2 + 2 * x_iter + 5
    x_iter = x_iter - yita * (2 * x_iter + 2)
    y_next = x_iter ** 2 + 2 * x_iter + 5
    plt.scatter(x_iter, y_last)  # 这行的作用是：把每一次迭代的 x 取值（x_iter）和对应的函数值（y_last）作为坐标点，画在图像上，
    if abs(y_next - y_last) < 1e-100:
        break
print('最小值点x=', x_iter, '最小值y', y_next, '迭代次数n=', count)
x = np.linspace(-4, 6, 100)
y = x ** 2 + 2 * x + 5
plt.plot(x, y, '--')
plt.show()

# 2.Newton法
def f(x):
    y=x**3-x-1#求根方程的表达式
    return y
def g(x):
    y=3*x**2-1#求根方程的导函数
    return y
def main():
    x_0=1.5 #取初值
    e=10**(-9) #误差要求
    L=0 #初始化迭代次数
    while L<100: #采用残差来判断
        x1=x_0-f(x_0)/g(x_0) #迭代公式,x(n+1)=x(n)-f(x(n))/f'(x(n))
        x_0=x1
        L=L+1 #统计迭代次数
        if abs(f(x_0)-0)<e:
            break
    print(f"x1={x1}") #输出数值解
    print(f(x_0)-0)  # 验证解的正确性
    print(f"L={L}") #输出迭代次数
if __name__ == '__main__':
   main()

   import numpy as np
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D


   # ===================== 1. 定义目标函数、梯度、海森矩阵 =====================
   # 目标函数：Rosenbrock函数（香蕉函数）
   def rosenbrock(x, y):
       return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


   # 梯度（一阶偏导数）
   def gradient(x, y):
       df_dx = -2 * (1 - x) - 400 * x * (y - x ** 2)  # 对x的偏导
       df_dy = 200 * (y - x ** 2)  # 对y的偏导
       return np.array([df_dx, df_dy])


   # 海森矩阵（二阶偏导数）
   def hessian(x, y):
       d2f_dx2 = 2 - 400 * (y - x ** 2) + 800 * x ** 2  # 二阶偏导∂²f/∂x²
       d2f_dxdy = -400 * x  # 混合偏导∂²f/∂x∂y
       d2f_dydx = -400 * x  # 混合偏导∂²f/∂y∂x
       d2f_dy2 = 200  # 二阶偏导∂²f/∂y²
       return np.array([[d2f_dx2, d2f_dxdy], [d2f_dydx, d2f_dy2]])


   # ===================== 2. 牛顿迭代法求解最小值 =====================
   def newton_iteration(initial_point, tol=1e-6, max_iter=100):
       """
       牛顿迭代法求解二元函数最小值
       :param initial_point: 初始点 [x0, y0]
       :param tol: 收敛阈值（梯度范数小于该值则停止）
       :param max_iter: 最大迭代次数
       :return: 最优解、迭代过程的点、迭代过程的函数值
       """
       x_k = np.array(initial_point, dtype=np.float64)  # 当前迭代点
       iter_points = [x_k.copy()]  # 记录迭代路径
       iter_values = [rosenbrock(*x_k)]  # 记录迭代过程的函数值

       for i in range(max_iter):
           grad = gradient(*x_k)
           # 收敛判断：梯度的L2范数小于阈值
           if np.linalg.norm(grad) < tol:
               print(f"迭代 {i + 1} 次后收敛")
               break
           # 计算海森矩阵并求逆
           H = hessian(*x_k)
           H_inv = np.linalg.inv(H)
           # 牛顿迭代更新
           x_k = x_k - H_inv @ grad
           # 记录迭代结果
           iter_points.append(x_k.copy())
           iter_values.append(rosenbrock(*x_k))

       # 若达到最大迭代次数仍未收敛
       if i == max_iter - 1 and np.linalg.norm(gradient(*x_k)) >= tol:
           print(f"达到最大迭代次数 {max_iter}，未完全收敛")

       return x_k, np.array(iter_points), np.array(iter_values)


   # ===================== 3. 可视化分析（支持中文，无变量未定义错误） =====================
   def visualize_results(iter_points, iter_values):
       # ========== 全局设置中文（核心：放在所有绘图代码之前） ==========
       plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows：黑体；Mac：['Arial Unicode MS']；Linux：['WenQuanYi Micro Hei']
       plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示为方块的问题

       # 生成网格数据
       x = np.linspace(-2, 2, 200)
       y = np.linspace(-1, 3, 200)
       X, Y = np.meshgrid(x, y)
       Z = rosenbrock(X, Y)

       # 创建画布和子图（先创建ax1/ax2/ax3，再调用其方法）
       fig = plt.figure(figsize=(18, 6))

       # 子图1：3D曲面（中文标签）
       ax1 = fig.add_subplot(131, projection='3d')  # 先定义ax1，再使用
       surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, antialiased=True)
       ax1.scatter(iter_points[:, 0], iter_points[:, 1], iter_values,
                   color='red', s=50, label='迭代点', zorder=5)
       ax1.set_xlabel('x轴')  # 无需额外传fontproperties，全局已生效
       ax1.set_ylabel('y轴')
       ax1.set_zlabel('z轴 = f(x,y)')
       ax1.set_title('Rosenbrock函数3D曲面 + 迭代路径')
       ax1.legend()
       fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

       # 子图2：等高线（中文标签）
       ax2 = fig.add_subplot(132)  # 先定义ax2，再使用
       contour = ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
       ax2.plot(iter_points[:, 0], iter_points[:, 1], 'r-o', linewidth=2, markersize=6, label='迭代路径')
       ax2.scatter(1, 1, color='black', s=100, marker='*', label='理论最小值点(1,1)')
       ax2.set_xlabel('x轴')
       ax2.set_ylabel('y轴')
       ax2.set_title('函数等高线 + 迭代路径')
       ax2.legend()
       plt.colorbar(contour, ax=ax2)

       # 子图3：收敛曲线（中文标签）
       ax3 = fig.add_subplot(133)  # 先定义ax3，再使用
       ax3.plot(range(len(iter_values)), iter_values, 'b-o', linewidth=2, markersize=6)
       ax3.set_xlabel('迭代次数')
       ax3.set_ylabel('函数值 f(x,y)')
       ax3.set_title('迭代收敛曲线')
       ax3.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()


   # ===================== 4. 主函数执行 =====================
   if __name__ == "__main__":
       # 初始点
       initial_point = [-1, 2]
       # 执行牛顿迭代
       optimal_point, iter_points, iter_values = newton_iteration(initial_point)

       # 输出结果
       print(f"初始点: {initial_point}")
       print(f"最优解 (x, y): {optimal_point.round(6)}")
       print(f"最优值 f(x,y): {rosenbrock(*optimal_point):.6f}")

       # 可视化（无变量未定义错误）
       visualize_results(iter_points, iter_values)
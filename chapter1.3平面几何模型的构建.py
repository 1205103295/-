# %% 第一章 解析方法与几何模型 - 1.3 平面几何模型构建
# 导入对应依赖包
import numpy as np
from scipy.optimize import fsolve
import pandas as pd

# %% 1.3 平面几何模型构建
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

# 定义表格
data = {"测地距中心点处的距离/m": [],
        "海水深度/m": [],
        "覆盖宽度/m": [],
        "与前一条侧线的重叠率": []}
df = pd.DataFrame(data)

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

eta = np.insert(eta, 0, 0)
# 在表格中输出数据
df["测地距中心点处的距离/m"] = [-800, -600, -400, -200, 0, 200, 400, 600, 800]
df["海水深度/m"] = D
df["覆盖宽度/m"] = W
df["与前一条侧线的重叠率"] = eta

print("表格数据：\n", df)

# 导出excel表格（注意：确保路径中的文件夹已存在，或补充文件夹创建逻辑）
save_path = r"D:\数模导论研习社\result.xlsx"  # r表示原生字符串，避免转义符问题
df.to_excel(excel_writer=save_path, index=False)  # index=False不导出行索引
print(f"文件已成功保存到：{save_path}")
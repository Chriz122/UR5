import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 讀取姿態資料
try:
    with open('a20241107a/20260208_162701/pose.txt', 'r') as f:
        data = np.loadtxt(f)
except FileNotFoundError:
    raise ValueError("無法開啟 pose.txt")

if data.shape[0] < 4:
    raise ValueError("資料少於 4 行")

# 提取角落
X_corners = data[:4, 0]
Y_corners = data[:4, 1]
Z_corners = data[:4, 2]

# 額外資料
extra_data = data[:4, 3:9]

# 網格
rows = 12
cols = 11
X_grid, Y_grid = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))

# X, Y, Z 的雙線性插值
X = ((1 - X_grid) * (1 - Y_grid) * X_corners[0] +
     X_grid * (1 - Y_grid) * X_corners[1] +
     X_grid * Y_grid * X_corners[3] +
     (1 - X_grid) * Y_grid * X_corners[2])

Y = ((1 - X_grid) * (1 - Y_grid) * Y_corners[0] +
     X_grid * (1 - Y_grid) * Y_corners[1] +
     X_grid * Y_grid * Y_corners[3] +
     (1 - X_grid) * Y_grid * Y_corners[2])

Z = ((1 - X_grid) * (1 - Y_grid) * Z_corners[0] +
     X_grid * (1 - Y_grid) * Z_corners[1] +
     X_grid * Y_grid * Z_corners[3] +
     (1 - X_grid) * Y_grid * Z_corners[2])

# 插值額外欄位
extra_cols = np.zeros((rows * cols, 6))
for i in range(6):
    extra_cols[:, i] = ((1 - X_grid.flatten()) * (1 - Y_grid.flatten()) * extra_data[0, i] +
                        X_grid.flatten() * (1 - Y_grid.flatten()) * extra_data[1, i] +
                        X_grid.flatten() * Y_grid.flatten() * extra_data[3, i] +
                        (1 - X_grid.flatten()) * Y_grid.flatten() * extra_data[2, i])

# 合併
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten(), extra_cols))

# 重塑並排列
points = points.reshape(rows, cols, 9)
points = np.transpose(points, (1, 0, 2))
points = points.reshape(-1, 9)

# 寫入 pose.txt
np.savetxt('a20241107a/pose.txt', points, fmt='%f')

print("資料寫入 pose.txt，包含 132 行和 9 欄")

# 繪製 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.plot_surface(points[:, 0].reshape(cols, rows).T,
                points[:, 1].reshape(cols, rows).T,
                points[:, 2].reshape(cols, rows).T)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('11x12 網格插值點 3D 繪圖')
plt.show()
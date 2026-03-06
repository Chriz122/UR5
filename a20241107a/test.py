# Fix mpl_toolkits version conflict - MUST be at very start before any imports
import sys

# Remove system dist-packages from path to avoid mpl_toolkits version mismatch
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

# Also remove any cached mpl_toolkits modules that were loaded by matplotlib
for mod in list(sys.modules.keys()):
    if 'mpl_toolkits' in mod:
        del sys.modules[mod]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 讀取姿態資料
try:
    with open('a20241107a/20260305_113053/pose.txt', 'r') as f:
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

# pose.txt 4個角點排列順序：左上[0], 右上[1], 左下[2], 右下[3]
# 網格
rows = 12   # 垂直方向角點數（上→下）
cols = 11   # 水平方向角點數（左→右）
#
# X_grid[r, c] = c/(cols-1)，0=左 → 1=右
# Y_grid[r, c] = r/(rows-1)，0=上 → 1=下
X_grid, Y_grid = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))

# X, Y, Z 雙線性插值（角點配對）：
#   (1-Xg)(1-Yg) → 左上 [0]
#   Xg*(1-Yg)   → 右上 [1]
#   (1-Xg)*Yg   → 左下 [2]
#   Xg*Yg       → 右下 [3]
X = ((1 - X_grid) * (1 - Y_grid) * X_corners[0] +
     X_grid       * (1 - Y_grid) * X_corners[1] +
     (1 - X_grid) * Y_grid       * X_corners[2] +
     X_grid       * Y_grid       * X_corners[3])

Y = ((1 - X_grid) * (1 - Y_grid) * Y_corners[0] +
     X_grid       * (1 - Y_grid) * Y_corners[1] +
     (1 - X_grid) * Y_grid       * Y_corners[2] +
     X_grid       * Y_grid       * Y_corners[3])

Z = ((1 - X_grid) * (1 - Y_grid) * Z_corners[0] +
     X_grid       * (1 - Y_grid) * Z_corners[1] +
     (1 - X_grid) * Y_grid       * Z_corners[2] +
     X_grid       * Y_grid       * Z_corners[3])

# 插值額外欄位（角度欄位 3~8）
extra_cols = np.zeros((rows * cols, 6))
for i in range(6):
    extra_cols[:, i] = ((1 - X_grid.flatten()) * (1 - Y_grid.flatten()) * extra_data[0, i] +
                        X_grid.flatten()       * (1 - Y_grid.flatten()) * extra_data[1, i] +
                        (1 - X_grid.flatten()) * Y_grid.flatten()       * extra_data[2, i] +
                        X_grid.flatten()       * Y_grid.flatten()       * extra_data[3, i])

# 合併（X.flatten() 是 row-major：從左到右掃完一行再換下一行）
# shape (rows=12, cols=11) → flatten → [r=0,c=0], [r=0,c=1], ..., [r=0,c=10], [r=1,c=0], ...
# 這與 cv2.findChessboardCorners 的輸出順序（從左到右、從上到下）完全一致
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten(), extra_cols))
# ⚠ 不做 np.transpose！原來的 transpose(1,0,2) 把輸出改為 col-major（先列後行），
#    與 cv2 順序相反，是錯誤的。

# 寫入 pose.txt
np.savetxt('a20241107a/pose.txt', points, fmt='%f')

print("資料寫入 pose.txt，包含 132 行和 9 欄")

# 驗證順序：第0點應為左上，第10點應為右上，第11點應為第2行左邊
print(f"點[0]   左上: X={points[0,0]*1000:.1f}mm Y={points[0,1]*1000:.1f}mm Z={points[0,2]*1000:.1f}mm")
print(f"點[10]  右上: X={points[10,0]*1000:.1f}mm Y={points[10,1]*1000:.1f}mm Z={points[10,2]*1000:.1f}mm")
print(f"點[11]  第2行左: X={points[11,0]*1000:.1f}mm Y={points[11,1]*1000:.1f}mm Z={points[11,2]*1000:.1f}mm")
print(f"點[131] 右下: X={points[131,0]*1000:.1f}mm Y={points[131,1]*1000:.1f}mm Z={points[131,2]*1000:.1f}mm")

# 繪製 3D（row-major，shape (rows,cols) 直接用）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.plot_surface(points[:, 0].reshape(rows, cols),
                points[:, 1].reshape(rows, cols),
                points[:, 2].reshape(rows, cols))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('11x12 網格插值點 3D 繪圖（從左→右、上→下）')
plt.show()
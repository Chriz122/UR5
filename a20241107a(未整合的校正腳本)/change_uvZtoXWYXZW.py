import cv2
import numpy as np
import matplotlib.pyplot as plt

# 清除主控台、變數，並關閉所有圖形
# 相當於 MATLAB 中的 clc, clear, close all

a = cv2.imread('A1.png')
if a is None:
    raise ValueError("無法載入影像 A1.png")

c = np.loadtxt('D1.txt')

# 檢測棋盤格點
# 假設棋盤格大小，例如 (11, 12) 用於 132 個點 (11*12=132)
board_size = (11, 12)  # 如有需要請調整
ret, corners = cv2.findChessboardCorners(a, board_size)
if not ret:
    raise ValueError("未找到棋盤格")

imagePoints = corners.reshape(-1, 2)  # 展平為 [u, v]

# h = [1:132]
h = np.arange(1, 133)  # 1 到 132
point = np.ones((len(h), 3))

ZZZ = np.ones(len(h))

plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
plt.show()

for n in range(len(h)):
    u = imagePoints[h[n]-1, 0]  # MATLAB 是 1 索引
    v = imagePoints[h[n]-1, 1]
    Z = c[int(round(v)), int(round(u))]
    ZZZ[n] = Z
    # P = [u*Z/10000; v*Z/10000; Z; 1]
    P = np.array([u * Z / 10000, v * Z / 10000, Z, 1])
    T = np.array([
        [-0.287948924281509, 16.5637329067818, -0.420194026689613, -46.7921574040553],
        [16.6041266797602, 0.497127857137068, -0.557910226216805, -560.943017045044],
        [0.491070106727183, -0.331688795630331, -1.03724117409923, 1217.81527516523],
        [0, 0, 0, 1]
    ])
    # 替代 T 已註解
    tt = np.zeros((len(h), 2))
    tt[n, 0] = u
    tt[n, 1] = v
    plt.text(tt[n, 0], tt[n, 1], 'o', color='g')
    # W = T * P / 1000
    W = T @ P / 1000
    point[n, :] = W[:3]

plt.show()

MIN = np.min(ZZZ)
MAX = np.max(ZZZ)

print(f"MIN: {MIN}")
print(f"MAX: {MAX}")
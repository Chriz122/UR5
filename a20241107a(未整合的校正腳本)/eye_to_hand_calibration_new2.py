import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 清除主控台、變數，並關閉所有圖形

point_num = 132
list_vals = np.arange(5, 10)  # [5:9]
NUM = len(list_vals)
impoint = np.ones((NUM * point_num, 4))
XYZ = np.zeros((NUM, 3))
cam_location = np.ones((NUM * point_num, 3))
a = np.zeros((132, 3))
XYZ_ = XYZ.copy()

# # 產生 a (模擬數據)
# for n in range(1, 12):  # 1 到 11
#     start = (n-1)*11 + 1
#     end = n*11 + 1
#     a[start-1:end-1, 0] = 0.025 * n

# for n in range(2, 12):  # 2 到 11
#     indices = np.arange(n-1, 132, 11)
#     a[indices, 1] = -0.025 * (n - 1)

# for n in range(1, 6):  # 1 到 5
#     start = (n-1) * point_num
#     end = n * point_num
#     cam_location[start:end, :3] = a * 1000
#     cam_location[start:end, 2] += 50 * (n - 1)

# 繪圖
# 假設 plotCalbPoint3d_noText 在其他地方定義
# plotCalbPoint3d_noText(cam_location)
# plt.view(-20, 90)
# plt.figure()

count = 0

for c in [5, 6, 7, 8, 9]:  # 對應 A5 到 A9
    count += 1
    pic_name = f'A{c}.bmp'
    I = cv2.imread(pic_name)
    if I is None:
        raise ValueError(f"Could not load {pic_name}")

    # 檢測棋盤格
    board_size = (11, 12)
    ret, corners = cv2.findChessboardCorners(I, board_size)
    if not ret:
        raise ValueError(f"未找到 {pic_name} 中的棋盤格")

    imagePoints = corners.reshape(-1, 2)

    # 建立網格
    x_mesh, y_mesh = np.meshgrid(np.arange(1, 641), np.arange(1, 481))  # 1:640, 1:480

    txt_name = f'D{c}.txt'
    depth = np.loadtxt(txt_name)

    # 尋找非零深度點
    valid = depth != 0
    x_valid = x_mesh[valid]
    y_valid = y_mesh[valid]
    depth_valid = depth[valid]

    # 插值
    imagePoints_z = griddata((x_valid, y_valid), depth_valid, (imagePoints[:, 0], imagePoints[:, 1]), method='linear')

    imagePoints = np.column_stack((imagePoints, imagePoints_z))

    # 設定 Z 為最小值
    imagePoints[:, 2] = np.min(imagePoints[:, 2])

    start = (count-1) * point_num
    end = count * point_num
    impoint[start:end, :3] = imagePoints

# 縮放
impoint[:, 0] = impoint[:, 0] * impoint[:, 2] / 10000
impoint[:, 1] = impoint[:, 1] * impoint[:, 2] / 10000

# 解 C
C = np.zeros((3, 4))
for a in range(3):
    C[a, :] = np.linalg.lstsq(impoint, cam_location[:, a], rcond=None)[0]

# 輸出轉移矩陣 C
print(C)

test = impoint @ C.T
tt = cam_location[:, :3]
t = (tt - test) ** 2
T = np.sqrt(np.sum(t, axis=1))

# plt.bar(np.arange(1, NUM*point_num + 1), T)
# plt.title('Calibration Point Distance Error')
# plt.xlabel('Point Index')
# plt.ylabel('Distance Error (mm)')
# plt.show()

# MSE_xyz = np.sum(t) / len(t)
# MSE_distErr = np.sum(T) / len(T)

# print(f"MSE_xyz: {MSE_xyz}")
# print(f"MSE_distErr: {MSE_distErr}")
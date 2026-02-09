# Fix mpl_toolkits version conflict - MUST be at very start before any imports
import sys
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]
for mod in list(sys.modules.keys()):
    if 'mpl_toolkits' in mod:
        del sys.modules[mod]

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Register 3D projection
from scipy.interpolate import griddata
from plotCalbPoint3d_noText import plotCalbPoint3d_noText

point_num = 132
list_vals = np.arange(5, 10)  # [5:9]
NUM = len(list_vals)
impoint = np.ones((NUM * point_num, 4))
XYZ = np.zeros((NUM, 3))
cam_location = np.ones((NUM * point_num, 9))
a = np.zeros((132, 3))
XYZ_ = XYZ.copy()

# Load pose data - use the INTERPOLATED pose.txt (132 points)
# NOT the raw 9-row file at 20260208_162701/pose.txt
try:
    pose_data = np.loadtxt('a20241107a/pose.txt')
    print(f"Loaded pose.txt with shape: {pose_data.shape}", flush=True)
except Exception as e:
    raise ValueError(f"Could not load pose.txt: {e}")

# Validate we have 132 points
if pose_data.shape[0] != point_num:
    raise ValueError(f"Expected {point_num} points in pose.txt, got {pose_data.shape[0]}")

# pose.txt format (9 columns):
# Col 0-2: X, Y, Z position in meters
# Col 3-5: Euler angles in degrees (NOT used for calibration, keep as-is)
# Col 6-8: Rotation vector in radians (rx, ry, rz)

# For calibration, we need world coordinates (cam_location) for each chessboard point
# Since we have the same 132-point grid for all images A5-A9, we tile the pose data

for n in range(1, NUM + 1):
    start_idx = (n-1) * point_num
    end_idx = n * point_num
    
    # Copy pose data to this image's slice
    cam_location[start_idx:end_idx, :] = pose_data.copy()
    
    # Convert ONLY position (cols 0-2) to mm, leave angles unchanged
    cam_location[start_idx:end_idx, 0:3] *= 1000  # X, Y, Z: meters -> mm
    # Columns 3-8 are angles, do NOT multiply by 1000!


# 繪圖
ax = plotCalbPoint3d_noText(cam_location)
if ax:
    ax.view_init(elev=-20, azim=90)
# plt.figure() # Creates an empty figure, unnecessary here


count = 0

for c in [5, 6, 7, 8, 9]:  # 對應 A5 到 A9
    count += 1
    pic_name = f'a20241107a/20260208_162701/A/A{c}.png'
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

    txt_name = f'a20241107a/20260208_162701/D/D{c}.txt'
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
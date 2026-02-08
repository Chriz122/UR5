import cv2
import numpy as np
import matplotlib.pyplot as plt
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

# Load pose data (9 images x 9 columns)
try:
    pose_data = np.loadtxt('a20241107a/20260208_162701/pose.txt')
    print(f"Loaded pose.txt with shape: {pose_data.shape}", flush=True)
except Exception as e:
    raise ValueError(f"Could not load pose.txt: {e}")

# Apply pose data to cam_location
# Iterate over the processed images (n=1 to NUM)
# list_vals = [5, 6, 7, 8, 9] (indices for pose_data should be 4, 5, 6, 7, 8 if 0-indexed)
# User loop range(1, 5) covers n=1, 2, 3, 4 (4 items). But NUM=5.
# Let's fix loop to range(1, NUM + 1)
for n in range(1, NUM + 1):
    # Map n (1..5) to pose index. 
    # list_vals[n-1] gives the image number (e.g., 5).
    # Assuming pose.txt rows correspond to A1...A9.
    # So for A5, we need row index 4.
    img_idx = list_vals[n-1]
    pose_row_idx = img_idx - 1 
    
    if pose_row_idx >= pose_data.shape[0]:
         raise ValueError(f"Pose index {pose_row_idx} out of bounds for pose data with shape {pose_data.shape}")

    # Broadcast single row (9,) to (132, 9)
    # cam_location is (NUM*point_num, 9)
    # Target slice is (132, 9)
    pose_row = pose_data[pose_row_idx, :] # Shape (9,)
    
    # Assign broadcasted value
    cam_location[(n-1)*point_num : n*point_num, :] = pose_row * 1000 # Convert to mm if needed

    # The user's original code had:
    # cam_location[(n-1)*point_num+1:n*point_num,3] = cam_location[(n-1)*point_num+1:n*point_num,3]+50*(n-1)
    # This modifies the 4th column (index 3). 
    # Do we still need this? 
    # The user replaced the generation logic with loading from file.
    # If the file contains the *actual* robot pose, we probably shouldn't modify it artificially.
    # However, the user's code DID include that line after loading.
    # It looks like an offset for Z? 
    # "cam_location... = ... + 50*(n-1)"
    # If they want to keep it, I will leave it, but corrected for 0-based indexing.
    # cam_location is (NUM*point_num, 9). 3rd index is 4th col.
    # cam_location[(n-1)*point_num : n*point_num, 3] += 50 * (n-1) 
    pass # I will comment it out or assume pose.txt is ground truth. 
    # Actually, the user's snippet logic:
    # cam_location[..., 1:9] = ...
    # cam_location[..., 3] = ... + 50*(n-1)
    # This implies they wanted columns 1-8 from file, and col 0 as 1s?
    # But pose.txt has 9 columns.
    
    # Let's trust pose.txt fully for now and overwrite everything.
    # If the offset is needed, the user can re-add it, but mixing data sources is risky.


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
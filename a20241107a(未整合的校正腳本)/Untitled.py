import numpy as np

point_num = 132
list_vals = np.arange(1, 5)  # [1:4]
NUM = len(list_vals)
cam_location = np.ones((NUM * point_num, 3))
a = np.zeros((132, 3))

for n in range(1, 12):  # 1 到 11
    start = (n-1)*11
    end = n*11
    a[start:end, 0] = 0.025 * n

for n in range(2, 12):  # 2 到 11
    indices = np.arange(n-1, 132, 11)
    a[indices, 1] = -0.025 * (n - 1)

for n in range(1, 5):  # 1 到 4
    start = (n-1) * point_num
    end = n * point_num
    cam_location[start:end, :3] = a * 1000
    cam_location[start:end, 2] += 50 * (n - 1)

# cam_location 現已設定
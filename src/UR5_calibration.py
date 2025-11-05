import urx
import time
from pathlib import Path
import os
import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化參數
C = 10 / 1000  # 預設移動步長

# 初始化機器人
rob = urx.Robot("192.168.1.101")
rob.set_tcp((0, 0, 0.1, 0, 0, 0))  # 設定工具中心點 (TCP)
rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.3)
pose = rob.getl()  # 取得當前機器人位置

# 初始化 RealSense 相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度影像
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色影像
profile = pipeline.start(config)

# 取得深度比例因子
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is:", depth_scale)

# 設定深度影像對齊到彩色影像
align_to = rs.stream.color
align = rs.align(align_to)

temp = 1  # 用於命名影像與檔案的計數器

try:
    # 確保資料夾存在
    os.makedirs(Path("data/txts"), exist_ok=True)
    with open(Path("data/txts/pose.txt"), "w") as s:
        while True:
            # 取得影像幀
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # 檢查影像幀是否有效
            if not aligned_depth_frame or not color_frame:
                continue

            # 轉換影像資料
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 處理鍵盤輸入
            key = cv2.waitKey(10)

            # 儲存影像與深度資料
            if key & 0xFF == ord("x"):
                os.makedirs("data/images", exist_ok=True)
                name = Path(f"data/images/A{temp}.png")
                cv2.imwrite(name, color_image)
                name = Path(f"data/txts/D{temp}.txt")
                with open(name, "w") as ss:
                    for a in range(480):
                        for b in range(640):
                            ss.write("%f " % depth_image[a][b])
                        ss.write("\n")
                print(temp)
                temp += 1
                s.write(f"{pose[0]} {pose[1]} {pose[2]} -90 0 135\n")

            # 顯示影像
            color_image = np.uint8(color_image)
            cv2.imshow("A", color_image)
            depth_image = np.uint8(depth_image)
            depth_image[depth_image != 0] = depth_image[depth_image != 0] % 200 + 55
            cv2.imshow("D", depth_image)

            # 調整步長
            if key == ord("J"):
                C=10/1000
                print(10)    
            if key == ord("K"):
                C=1/1000
                print(1)
            if key == ord("L"):
                C=0.1/1000
                print(0.1)

            # 顯示當前機器人位置
            if key & 0xFF == ord("z"):
                pose = rob.getl()
                print(pose[0], pose[1], pose[2])

            # 控制機器人移動
            if key & 0xFF == ord("w"):  # 向上移動
                pose[2] += C
                rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)
            if key & 0xFF == ord("s"):  # 向下移動
                pose[2] -= C
                rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)
            if key & 0xFF == ord("a"):  # 向左移動
                pose[1] += C
                rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)
            if key & 0xFF == ord("d"):  # 向右移動
                pose[1] -= C
                rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)
            if key & 0xFF == ord("o"):  # 向前移動
                pose[0] += C
                rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)
            if key & 0xFF == ord("p"):  # 向後移動
                pose[0] -= C
                rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)

            # 結束程式
            if key & 0xFF == ord("q") or key == 27:  # "q" 或 ESC
                cv2.destroyAllWindows()
                break
finally:
    pipeline.stop()

# 移動機器人到最後位置
rob.movel((pose[0], pose[1], pose[2], 2.4184, -2.4184, 2.4184), 1, 0.1)


import sys
import os

# Ensure we use the virtual environment's site-packages first
venv_site_packages = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'lib', 'python3.10', 'site-packages')
if os.path.exists(venv_site_packages):
    sys.path.insert(0, venv_site_packages)

import urx
from urx.urrobot import RobotException
import socket
import cv2
import time
from math import *
import numpy as np
import pyrealsense2 as rs
from function_arm import rv2rpy, arm_movej
from algorithm.ant import create_distance_matrix, ant_colony_optimization
from orchid_pose_d435 import orchid_pose_seg_area_leafs_number_predict_d435_new

# NOTE: q/Q:開始, x:拍照, X:紀錄當前座標, p: use yolo to predict and move to the target

#-16.6151988924574	-1.35365000913771	0.581205593619286	-41.9357365124314
#-1.40063398542737	16.5840344888449	-0.360032069471067	-623.014017362210
#-0.294201422424942	0.0206102811194019	-1.04738461699090	1179.61724461631

C44_eyetohand4 = np.array([[-6.30842589e-07,  6.11072692e-08, -6.90856672e-05, -1.06794210e+02],
                            [ 2.97032996e-06,  1.12525001e-07,  1.18046337e-04, -2.94974979e+02],
                            [ 2.76420612e-06,  1.71288098e-07,  1.25929049e-04,  5.04399064e+02],
                            [0, 0, 0, 1]]) # D435i 轉移矩陣 eye to hand

MODEL = {
    "POSE": "models/best_all_0_degree_small.v2i.v11l_pose.pt",
    "SEG": "models/best_Yat-sen_University_orchid-idea.v7i.v11s_seg.pt"
}
UR5 = {
    "IP": "192.168.1.101",
    "PORT": 30001
}

def OPEN():
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((UR5["IP"],UR5["PORT"]))
    
    f = open(r'src/open.txt')
    text = []
    for line in f:
        text.append(line)
    strr = "".join(text)
    
    s.send(str.encode(strr))
    time.sleep(0.5)
    
    s.close()
    time.sleep(0.5)
    
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))
    
    print('open')
    
def CLOSE(width):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((UR5["IP"],UR5["PORT"]))
    
    f = open(r'src/close.txt')
    text = []
    for line in f:
        if "rq_set_pos_norm(100, " in line: # 8.5cm = 0.085 m
            pos = abs(round((width)/0.00085) - 100)
            line = "    rq_set_pos_norm(" + str(pos) + ', "1")\n'
            text.append(line)
        else:    
            text.append(line)
    
    strr = "".join(text)
    
    s.send(str.encode(strr))
    time.sleep(0.5)
    
    s.close()
    time.sleep(0.5)
    
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))

    print('close')
    
# t=0
C=10/1000
acc = 0.1
vel = 0.3
rob = urx.Robot(UR5["IP"])

rob.set_tcp((0, 0, 0.1, 0, 0, 0))

rob.set_payload(2, (0, 0, 0.1))
time.sleep(0.2)

pose = rob.getl()
posej = rob.getj()
count_intel=0
LEFT = 0
RIGHT = 0
FRONT = 0
BACK = 0
UP = 0
DOWN = 0
MOVE = 0

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
temp=1 #從多少開始
# Streaming loop

DIR_NAME = time.strftime("%Y%m%d_%H%M%S")
predict_pose_number = 1

if not os.path.isdir("data/" + DIR_NAME):
    os.makedirs("data/" + DIR_NAME)

try:
    with open('data/' + DIR_NAME  + '/pose.txt','w') as s:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image
    
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
    
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
    
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            key = cv2.waitKeyEx(10)
            # 對深度图黑洞區域進行填補
            hole_filling = rs.hole_filling_filter()
            filled_depth = hole_filling.process(aligned_depth_frame)   
            depth_frame_modify = np.asanyarray(filled_depth.get_data())
            colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame_modify, alpha=0.03), cv2.COLORMAP_JET)
            
            depth_image = depth_frame_modify
            color_image = np.asanyarray(color_frame.get_data())    
          
            color_image=np.uint8(color_image)
            color_image_copy = color_image.copy()
            
            # depth_image=np.uint8(depth_image)
            # depth_image[depth_image!=0]=depth_image[depth_image!=0]%200+55
            
            dc_images = np.hstack((color_image, colorized_depth))
            cv2.imshow('image and depth', dc_images)         

            if key & 0xFF == ord('J') or key & 0xFF == ord('j'):#按鍵J
                C=10/1000
                print(10)    
            if key & 0xFF == ord('K') or key & 0xFF == ord('k'):#按鍵K
                C=1/1000
                print(1)
            if key & 0xFF == ord('L') or key & 0xFF == ord('l'):#按鍵L
                C=0.1/1000
                print(0.1)
                       
            if key & 0xFF == ord('x') or key & 0xFF == ord('X'): # 純拍照
                
                posel = rob.getl()
                
                if not os.path.isdir("data/" + DIR_NAME + "/A"):
                    os.makedirs("data/" + DIR_NAME + "/A")
                    
                if not os.path.isdir("data/" + DIR_NAME + "/D"):
                    os.makedirs("data/" + DIR_NAME + "/D")    
                    
                name='data/' + DIR_NAME  + '/A/A' + str(temp) + '.png'
                cv2.imwrite(name,color_image)
                name='data/' + DIR_NAME  + '/D/D' + str(temp) + '.txt'
                with open(name,'w') as ss:
                    for a in range(480):
                        for b in range(640):
                            ss.write('%f ' %depth_image[a][b])
                        ss.write('\n')
                print(temp)
                temp+=1            
                color_image=np.uint8(color_image)
            
                dc_images = np.hstack((color_image, colorized_depth))
                cv2.imshow('AD', dc_images)
                
            # if key & 0xFF == ord('X') or key & 0xFF == ord('x'): # 純存基座座標
                
                posel = rob.getl()
                # print(temp)
                s.write('%f ' %posel[0])
                s.write('%f ' %posel[1])
                s.write('%f ' %posel[2])

                rpy = rv2rpy(posel[3],posel[4],posel[5])
                
                s.write('%f ' %rpy[0])
                s.write('%f ' %rpy[1])
                s.write('%f ' %rpy[2])
                
                s.write('%f ' %posel[3])
                s.write('%f ' %posel[4])
                s.write('%f ' %posel[5])
                
                s.write('\n')
                
                dl3 = degrees(posel[3]); dl4 = degrees(posel[4]); dl5 = degrees(posel[5]);
                
                print("當前位置和姿態（米和弧度）:")
                print(f"X: {posel[0]*1000:.2f} mm")
                print(f"Y: {posel[1]*1000:.2f} mm")
                print(f"Z: {posel[2]*1000-400:.2f} mm") # Z 要扣除 400 mm
                print(f"RX: {posel[3]:.3f} rad, {dl3:.3f}°")
                print(f"RY: {posel[4]:.3f} rad, {dl4:.3f}°")
                print(f"RZ: {posel[5]:.3f} rad, {dl5:.3f}°")
            
            if key & 0xFF == ord('8'): #向上
                pose[2]=pose[2]+C
                # pose=rob.getl()
                try:
                    rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                except RobotException as e:
                    print(f"Move failed: {e}")
                UP += 1
                print('UP: ', UP)
                
            if key & 0xFF == ord('2'): #向下
                pose[2]=pose[2]-C
                # pose=rob.getl()
                try:
                    rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                except RobotException as e:
                    print(f"Move failed: {e}")
                DOWN += 1
                print('DOWN: ', DOWN)
                
            if key & 0xFF == ord('w'): #向前
                pose[1]=pose[1]-C
                # pose=rob.getl()
                try:
                    rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                except RobotException as e:
                    print(f"Move failed: {e}")
                FRONT += 1
                print('FRONT: ', FRONT)
                
            if key & 0xFF == ord('s'): #向後
                pose[1]=pose[1]+C
                # pose=rob.getl()
                try:
                    rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                except RobotException as e:
                    print(f"Move failed: {e}")
                BACK += 1
                print('BACK: ', BACK)

            if key & 0xFF == ord('d'): #向右
                pose[0]=pose[0]-C
                # pose=rob.getl()
                try:
                    rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                except RobotException as e:
                    print(f"Move failed: {e}")
                RIGHT += 1
                print('RIGHT: ', RIGHT)
                
            if key & 0xFF == ord('a'): #向左
                pose[0]=pose[0]+C
                # pose=rob.getl()
                try:
                    rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                except RobotException as e:
                    print(f"Move failed: {e}")
                LEFT += 1   
                print('LEFT: ', LEFT)
                
            if (key & 0xFF == ord('q')) or (key & 0xFF == ord('Q')): 
                arm_movej((radians(-90.07),
                            radians(-70.8),
                            radians(-78.95),             # 負的向下
                            radians(-120.25),
                            radians(90),
                            radians(0.39)),acc,vel)  #校正角度
                pose = rob.getl()
                print(pose[0],pose[1],pose[2])
                print("read pose")
                                                    
            if key & 0xFF == ord('p'):                
                ALL_results_rows, img, csv_data, img_name, predict_time, angle_time, seg_time = orchid_pose_seg_area_leafs_number_predict_d435_new(color_image, depth_image, MODEL["POSE"], MODEL["SEG"], 
                                                                                                                                               predict_pose_number)
                
                if ALL_results_rows is None or len(ALL_results_rows) == 0:
                    print("No valid pose data detected.")
                    
                else:  
                    csv_data.append([])
                    csv_data.append(["Predict-time(sec):", predict_time])
                    csv_data.append(["Angle-time(sec):", angle_time])
                    csv_data.append(["Segment-time(sec):", seg_time])
                    # 创建距离矩阵
                    distance_matrix = create_distance_matrix(ALL_results_rows)
                    
                    # 使用蚁群算法规划路线
                    best_route_indices, best_distance, ant_time = ant_colony_optimization(distance_matrix)
                    csv_data.append(["Ant Path-execute-time(sec):", ant_time])
                    
                    best_route = [ALL_results_rows[i] for i in best_route_indices]
                    
                    # 打印总距离
                    print(f"总距离: {best_distance}")
                    
                    # cv2.imshow("windows_name", img)
                    cv2.imwrite("data/" + DIR_NAME + "/" + img_name, img) # 轉前辨識圖
                    # 绘制路径
                    for i in range(len(best_route)):
                        loc = best_route[i]
                        # # print(loc)
                        # XYZ_temp = [loc[2][1][0], loc[2][1][1], depth_image[loc[2][1][1], loc[2][1][0]], 1]
                        XYZ_temp = [loc[1][0], loc[1][1], depth_image[loc[1][1], loc[1][0]], 1]
                        XYZ_temp = np.array(XYZ_temp,'float32')
                        XYZ_temp[0] = XYZ_temp[0] * XYZ_temp[2]/10000
                        XYZ_temp[1] = XYZ_temp[1] * XYZ_temp[2]/10000
                        B_temp = C44_eyetohand4.dot(XYZ_temp)
                        B_temp = B_temp/1000
                        
                        if i > 0:
                            prev_loc = best_route[i-1]
                            cv2.line(img, tuple(prev_loc[1]), tuple(loc[1]), (255, 0, 255), 2)
                        
                        rob.movel((B_temp[0], B_temp[1], 0.484763, 0, 3.1271, 0), 1, 0.1) # 移動到蘭花中心
                    
                        if loc[3][0]+90 < 0:
                            # 夾爪順時針轉
                            CLOSE((loc[3][2]+12)/1000)
                            print(f'夾爪順時針轉{abs(loc[3][0]+90)}度') # 夾爪要對齊植株介質的角度
                            posej = rob.getj()
                            posej[5] = posej[5] + radians(abs(loc[3][0]+90))
                            rob.movej((posej[0],
                                              posej[1],
                                              posej[2],
                                              posej[3],
                                              posej[4],
                                              posej[5]),acc,vel)  #校正角度
                            # time.sleep(5)
                            posel_R = rob.getl()
                            rob.movel((B_temp[0], B_temp[1], 0.375, posel_R[3], posel_R[4], posel_R[5]), 1, 0.1) # 插下去
                            
                            if loc[4] < 0:   
                                print(f'夾爪逆時針轉{abs(loc[4])}度') # 夾爪要轉動植株的角度
                                posej = rob.getj()
                                posej[5] = posej[5] - radians(abs(loc[4]))
                                rob.movej((posej[0],
                                            posej[1],
                                            posej[2],
                                            posej[3],
                                            posej[4],
                                            posej[5]),acc,vel)  #校正角度
                                # time.sleep(5)
                                posel_R2 = rob.getl()
                                rob.movel((B_temp[0], B_temp[1], 0.484763, posel_R2[3], posel_R2[4], posel_R2[5]), 1, 0.1) # 拔出來
                                rob.movel((B_temp[0], B_temp[1], 0.484763, 0, 3.1271, 0), 1, 0.1) # 歸位
                                
                            else:
                                print(f'夾爪順時針轉{abs(loc[4])}度') # 夾爪要轉動植株的角度
                                posej = rob.getj()
                                posej[5] = posej[5] + radians(abs(loc[4]))
                                rob.movej((posej[0],
                                            posej[1],
                                            posej[2],
                                            posej[3],
                                            posej[4],
                                            posej[5]),acc,vel)  #校正角度
                                # time.sleep(5)
                                posel_R2 = rob.getl()
                                rob.movel((B_temp[0], B_temp[1], 0.484763, posel_R2[3], posel_R2[4], posel_R2[5]), 1, 0.1)
                                rob.movel((B_temp[0], B_temp[1], 0.484763, 0, 3.1271, 0), 1, 0.1)
                            
                        elif loc[3][0]+90 > 0:
                            # 夾爪逆時針轉
                            CLOSE((loc[3][2]+12)/1000)
                            print(f'夾爪逆時針轉{abs(loc[3][0]+90)}度') # 夾爪要對齊植株介質的角度
                            posej = rob.getj()
                            posej[5] = posej[5] - radians(abs(loc[3][0]+90))
                            rob.movej((posej[0],
                                        posej[1],
                                        posej[2],
                                        posej[3],
                                        posej[4],
                                        posej[5]),acc,vel)  #校正角度
                            # time.sleep(5)
                            posel_R = rob.getl()
                            rob.movel((B_temp[0], B_temp[1], 0.375, posel_R[3], posel_R[4], posel_R[5]), 1, 0.1)
                            
                            if loc[4] < 0:   
                                print(f'夾爪逆時針轉{abs(loc[4])}度') # 夾爪要轉動植株的角度
                                posej = rob.getj()
                                posej[5] = posej[5] - radians(abs(loc[4]))
                                rob.movej((posej[0],
                                                  posej[1],
                                                  posej[2],
                                                  posej[3],
                                                  posej[4],
                                                  posej[5]),acc,vel)  #校正角度
                                # time.sleep(5)
                                posel_R2 = rob.getl()
                                rob.movel((B_temp[0], B_temp[1], 0.484763, posel_R2[3], posel_R2[4], posel_R2[5]), 1, 0.1)
                                rob.movel((B_temp[0], B_temp[1], 0.484763, 0, 3.1271, 0), 1, 0.1)
                                
                            else:
                                print(f'夾爪順時針轉{abs(loc[4])}度') # 夾爪要轉動植株的角度   
                                posej = rob.getj()
                                posej[5] = posej[5] + radians(abs(loc[4]))
                                rob.movej((posej[0],
                                                  posej[1],
                                                  posej[2],
                                                  posej[3],
                                                  posej[4],
                                                  posej[5]),acc,vel)  #校正角度
                                # time.sleep(5)
                                posel_R2 = rob.getl()
                                rob.movel((B_temp[0], B_temp[1], 0.484763, posel_R2[3], posel_R2[4], posel_R2[5]), 1, 0.1)
                                rob.movel((B_temp[0], B_temp[1], 0.484763, 0, 3.1271, 0), 1, 0.1)
                    
                    arm_movej((radians(-90.07),
                                radians(-70.8),
                                radians(-78.95),             # 負的向下
                                radians(-120.25),
                                radians(90),
                                radians(0.39)),acc,vel)  #校正角度
                    
                    time.sleep(1)
                                  
                    predict_pose_number += 1

            if key == 27: #按下 Esc 關閉視窗
                cv2.destroyAllWindows()
                rob.close()
                sys.exit()
                break
finally:
    pipeline.stop()
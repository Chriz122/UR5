import urx
import socket
from math import *
import threading
import function_intelFORyolo as f_intel
import os
import cv2
from function_arm import rv2rpy
import function_arm as f_arm
import numpy as np
import random
from find_WORLD import find_WORLD_eyetohand
import os
import cv2
import csv
import math
import time
from ultralytics import YOLO

from algorithm.ant import create_distance_matrix, ant_colony_optimization
from orchid_pose_d435 import orchid_pose_seg_area_leafs_number_predict_d435, orchid_pose_seg_area_leafs_number_predict_d435_new


# TODO: 將ASCII改ord('')，l功能可以拔掉
# NOTE: Q:開始, x:拍照, X:紀錄當前座標

#-16.6151988924574	-1.35365000913771	0.581205593619286	-41.9357365124314
#-1.40063398542737	16.5840344888449	-0.360032069471067	-623.014017362210
#-0.294201422424942	0.0206102811194019	-1.04738461699090	1179.61724461631


C44_eyetohand4 = np.array([[-16.6358891363565,-1.25258697737028,0.556322250640770,-47.8888839900582],
                [-1.13868096832108,16.6410828171158,-0.363319442237862,-629.383608102392],
                [-0.180824295282988,0.109614597224694,-1.05654307600949,1181.95724761074],
                [0, 0, 0, 1]]) # D435i 轉移矩陣 eye to hand


def OPEN():
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(("192.168.1.101",30001))
    
    f = open(r'open.txt')
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
    s.connect(("192.168.1.101", 30001))
    
    f = open(r'close.txt')
    text = []
    for line in f:
        if "rq_set_pos_norm(100, " in line: # 8.5cm = 0.085 m
            # pos = abs(round(width/0.00085) - 100)
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
rob = urx.Robot("192.168.1.101")

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




import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering

        


# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
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

skeleton = [[2, 1], [3, 1], [4, 1], [5, 1]]

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [255, 255, 255], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)

# kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

kpt_color  = pose_palette[[10, 0, 9, 7, 16]]

limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

DIR_NAME = "a20250411b" # 資料夾名稱
predict_pose_number = 1
I = 0 # 辨識名稱

if not os.path.isdir("D:/" + DIR_NAME):
    os.makedirs("D:/" + DIR_NAME)

try:
    with open('D:\\' + DIR_NAME  + '\\pose.txt','w') as s:
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
            cv2.imshow('AD', dc_images)
            
            # cv2.imshow('A', color_image)
            # cv2.imshow('D', colorized_depth)
            

            if key == 106:#按鍵J
                C=10/1000
                print(10)    
            if key == 107:#按鍵K
                C=1/1000
                print(1)
            if key == 108:#按鍵L
                C=0.1/1000
                print(0.1)
            
            #面朝右角度2.4184,2.4184,-2.4184   
            #面朝前角度0,2.2214,-2.2214
            
            if key & 0xFF == ord('x'): # 純拍照
                
                posel = rob.getl()
                
                if not os.path.isdir("D:/" + DIR_NAME + "/A"):
                    os.makedirs("D:/" + DIR_NAME + "/A")
                    
                if not os.path.isdir("D:/" + DIR_NAME + "/D"):
                    os.makedirs("D:/" + DIR_NAME + "/D")    
                    
                name='D:\\' + DIR_NAME  + '\\A\\A' + str(temp) + '.bmp'
                cv2.imwrite(name,color_image)
                name='D:\\' + DIR_NAME  + '\\D\\D' + str(temp) + '.txt'
                with open(name,'w') as ss:
                    for a in range(480):
                        for b in range(640):
                            ss.write('%f ' %depth_image[a][b])
                        ss.write('\n')
                print(temp)
                temp+=1
                # s.write('%f ' %pose[0])
                # s.write('%f ' %pose[1])
                # s.write('%f ' %pose[2])

                # rpy = rv2rpy(pose[3],pose[4],pose[5])
                
                # s.write('%f ' %rpy[0])
                # s.write('%f ' %rpy[1])
                # s.write('%f ' %rpy[2])
                
                # s.write('%f ' %pose[3])
                # s.write('%f ' %pose[4])
                # s.write('%f ' %pose[5])
                
                # s.write('\n')
                
                # dl3 = degrees(posel[3]); dl4 = degrees(posel[4]); dl5 = degrees(posel[5]);
                
                # print("當前位置和姿態（米和弧度）:")
                # print(f"X: {posel[0]*1000:.2f} mm")
                # print(f"Y: {posel[1]*1000:.2f} mm")
                # print(f"Z: {posel[2]*1000-400:.2f} mm") # Z 要扣除 400 mm
                # print(f"RX: {posel[3]:.3f} rad, {dl3:.3f}°")
                # print(f"RY: {posel[4]:.3f} rad, {dl4:.3f}°")
                # print(f"RZ: {posel[5]:.3f} rad, {dl5:.3f}°")
            
                color_image=np.uint8(color_image)
                # depth_image=np.uint8(depth_image)
                # depth_image[depth_image!=0]=depth_image[depth_image!=0]%200+55
            
                dc_images = np.hstack((color_image, colorized_depth))
                cv2.imshow('AD', dc_images)
            
                # cv2.imshow('A', color_image)
                # cv2.imshow('D', colorized_depth)
                
            if key & 0xFF == ord('X'): # 純存基座座標
                
                posel = rob.getl()
                
                # if not os.path.isdir("D:/" + DIR_NAME + "/A"):
                #     os.makedirs("D:/" + DIR_NAME + "/A")
                    
                # if not os.path.isdir("D:/" + DIR_NAME + "/D"):
                #     os.makedirs("D:/" + DIR_NAME + "/D")    
                    
                # name='D:\\' + DIR_NAME  + '\\A' + str(temp) + '.bmp'
                # cv2.imwrite(name,color_image)
                # name='D:\\' + DIR_NAME  + '\\D' + str(temp) + '.txt'
                # with open(name,'w') as ss:
                #     for a in range(480):
                #         for b in range(640):
                #             ss.write('%f ' %depth_image[a][b])
                #         ss.write('\n')
                print(temp)
                # temp+=1
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
                
                # # 获取当前工具的位置（返回的是一个4x4的转换矩阵）
                # trans = rob.get_pose()  # This returns the transformation matrix from the base to the TCP
                # translation = trans.pos
                # rotation = trans.orient
         
                # print("trans:",trans)
                # print("Translation:", translation)
                # print("Rotation:", rotation)
   
                # 棋盘格设置
                # chessboard_size = (5, 3)
                # square_size = 0.025  # 每个方块的真实尺寸，单位为米

                # # 准备棋盘格三维点的坐标
                # objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
                # objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
                # objp *= square_size
                
                # # 转换为灰度图像
                # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('Detected Chessboard', gray_image)
                
                # # Color-segmentation to get binary mask
                # lwr = np.array([0, 0, 143])
                # upr = np.array([179, 61, 252])
                # hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                # msk = cv2.inRange(hsv, lwr, upr)
                
                # # Extract chess-board
                # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
                # dlt = cv2.dilate(msk, krn, iterations=5)
                # res = 255 - cv2.bitwise_and(dlt, msk)
                
                # # Displaying chess-board features
                # res = np.uint8(res)
                
                # cv2.imshow('Detected Chessboard', res)

                # 寻找棋盘格角点
                # ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
                # print(ret)
                # if ret:
                #     # 角点精细化
                #     corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), 
                #                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    
                #     # 绘制并显示角点
                #     color_image = cv2.drawChessboardCorners(color_image, chessboard_size, corners2, ret)
                #     cv2.imshow('Detected Chessboard', color_image)
                    
                #     # 获取相机内参和畸变系数
                #     profile = pipeline.get_active_profile()
                #     intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    
                #     # 将像素坐标转换为世界坐标
                #     world_points = []
                #     for corner in corners2:
                #         x, y = corner.ravel()
                #         depth =  depth_image[int(x), int(y)]
                #         if depth > 0:  # 有效深度值
                #             world_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                #             world_points.append(world_point)
    
                #     print("World coordinates of corners:", world_points)
                
            if key & 0xFF == ord('f'): 
                find_WORLD_eyetohand(4, 7, 'D:\\' + DIR_NAME  + '\\A\\A' + str(temp-1) + '.bmp', 
                                     'D:\\' + DIR_NAME  + '\\D\\D' + str(temp-1) + '.txt')
            
            if key == 2490368: #向上
                pose[2]=pose[2]+C
                # pose=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                UP += 1
                print('UP: ', UP)
                
            if key == 2621440: #向下
                pose[2]=pose[2]-C
                # pose=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                DOWN += 1
                print('DOWN: ', DOWN)
                
            if key & 0xFF == ord('w'): #向前
                pose[1]=pose[1]-C
                # pose=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                FRONT += 1
                print('FRONT: ', FRONT)
 
                
            if key & 0xFF == ord('s'): #向後
                pose[1]=pose[1]+C
                # pose=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                BACK += 1
                print('BACK: ', BACK)

                
            if key & 0xFF == ord('d'): #向右
                pose[0]=pose[0]-C
                # pose=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                RIGHT += 1
                print('RIGHT: ', RIGHT)
                
            if key  & 0xFF == ord('a'): #向左
                pose[0]=pose[0]+C
                # pose=rob.getl()
                rob.movel((pose[0],pose[1],pose[2],0, 3.1271, 0),1,0.1)
                LEFT += 1
                print('LEFT: ', LEFT)
                
            if key  & 0xFF == ord('1'): #頂部順時針轉
                posej = rob.getj()
                posej[5] = posej[5] + radians(90)
                rob.movej((posej[0],
                                  posej[1],
                                  posej[2],
                                  posej[3],
                                  posej[4],
                                  posej[5]),acc,vel)  #校正角度
                
                # print('LEFT: ', LEFT)
                
            if key  & 0xFF == ord('2'): #頂部逆時針轉
                posej = rob.getj()
                posej[5] = posej[5] - radians(90)
                rob.movej((posej[0],
                                  posej[1],
                                  posej[2],
                                  posej[3],
                                  posej[4],
                                  posej[5]),acc,vel)  #校正角度
                # print('LEFT: ', LEFT)
                
                
            if key & 0xFF == ord('Q'): #回到校正角度
                # f_arm.arm_movej((radians(-90),
                #                   radians(-127.5),
                #                   radians(-30),             # 負的向下
                #                   radians(-110.66),
                #                   radians(90),
                #                   radians(0)),acc,vel)  #校正角度
                # pose=rob.getl()
                
                # rob.movel((-0.06665594686972268, -0.4917156723613998, 0.48477046529675194, -0.6768943373881493, 3.0532656583809974, -0.004952265820574974),1,0.1)
                
                f_arm.arm_movej((radians(-90.07),
                                  radians(-70.8),
                                  radians(-78.95),             # 負的向下
                                  radians(-120.25),
                                  radians(90),
                                  radians(0.39)),acc,vel)  #校正角度
                # rob.movel((pose[0],pose[1],0.334763,0, 3.1271, 0),1,0.1)
                pose = rob.getl()
                print(pose[0],pose[1],pose[2])
                
            if key & 0xFF == ord('m'): # 關夾爪
                CLOSE(0.049) # 輸入物體直徑(單位:公尺(m))
                
            
            if key & 0xFF == ord('n'): # 開夾爪
                OPEN()

                
                
            if key  & 0xFF == ord('o'):
                X,Y,Z,ppp,kind=f_intel.getXYZT(color_image,depth_image)
                if len(kind) != 0:
                    pose=f_arm.arm_getl()
                                       
                    # print(X)
                    # print(Y)
                    # print(Z)
                    # print(ppp)
                    
                    f_arm.arm(X,Y,Z,kind,pose,acc,vel)
                    
                    num=len(kind)
                    for a in range(num):
                        rob.movel_tool((X[a],Y[a],Z[a],0,0,0),acc,vel)
                        time.sleep(1)
                        rob.movel_tool((-X[a],-Y[a],-Z[a],0,0,0),acc,vel)
                        time.sleep(1)
                        
            if key & 0xFF == ord('l'):
                # pose=rob.getl()
                rob.movel((-0.345454541275785,	-0.410412116471500, 0.334763,0, 3.1271, 0),1,0.1)
                
            '''   
            if key & 0xFF == ord('p'):
                X, Y, W, H = 86, 41, 386, 436
                # roi = color_image[Y:Y+H, X:X+W]
                pose_model_name = "best_all_0_degree_small.v2i.v11l_pose.pt"
                seg_model_name = "best_Yat-sen_University_orchid-idea.v7i.v11s_seg.pt"
                # ALL_results_rows, img, csv_data, img_name, predict_time, angle_time = orchid_pose_predict_d435(color_image, depth_image, pose_model_name, predict_pose_number)
                # ALL_results_rows, img, csv_data, img_name, predict_time, angle_time, seg_time = orchid_pose_seg_area_leafs_number_predict_d435(color_image, depth_image, pose_model_name, seg_model_name, 
                #                                                                                                                                predict_pose_number)
                
                ALL_results_rows, img, csv_data, img_name, predict_time, angle_time, seg_time = orchid_pose_seg_area_leafs_number_predict_d435_new(color_image, depth_image, pose_model_name, seg_model_name, 
                                                                                                                                               predict_pose_number)
                
                if ALL_results_rows is None or len(ALL_results_rows) == 0:
                    print("No valid pose data detected.")
                    
                else:  
                    cv2.imwrite("D:/" + DIR_NAME + "/" + "Before_" + str(predict_pose_number) + ".jpg", color_image_copy) # 轉前原圖
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
                    cv2.imwrite("D:/" + DIR_NAME + "/" + img_name, img) # 轉前辨識圖
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
                        
                        # cv2.imwrite(img_name, img)
                        # print("save done") 
                        # print(f"CSV file 'keypoints_data_{str(predict_pose_number)}.csv' has been saved.")
                        
                        # time.sleep(5)
                        # print('wait')                
            
                        # print(B_temp)
                        
                        if i > 0:
                            prev_loc = best_route[i-1]
                            cv2.line(img, tuple(prev_loc[1]), tuple(loc[1]), (255, 0, 255), 2)
                        
                        rob.movel((B_temp[0], B_temp[1], 0.484763, 0, 3.1271, 0), 1, 0.1) # 移動到蘭花中心
                    
                        if loc[3][0] < 0:
                            # 夾爪順時針轉
                            CLOSE((loc[3][2]+12)/1000)
                            print(f'夾爪順時針轉{abs(loc[3][0])}度') # 夾爪要對齊植株介質的角度
                            posej = rob.getj()
                            posej[5] = posej[5] + radians(abs(loc[3][0]))
                            rob.movej((posej[0],
                                              posej[1],
                                              posej[2],
                                              posej[3],
                                              posej[4],
                                              posej[5]),acc,vel)  #校正角度
                            # time.sleep(5)
                            posel_R = rob.getl()
                            rob.movel((B_temp[0], B_temp[1], 0.375, posel_R[3], posel_R[4], posel_R[5]), 1, 0.1) # 插下去
                            # posel = rob.getl()
                            # print(posel)
                            
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
                            
                        elif loc[3][0] > 0:
                            # 夾爪逆時針轉
                            CLOSE((loc[3][2]+12)/1000)
                            print(f'夾爪逆時針轉{abs(loc[3][0])}度') # 夾爪要對齊植株介質的角度
                            posej = rob.getj()
                            posej[5] = posej[5] - radians(abs(loc[3][0]))
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
                        
                    #--------------------------------------------------------------------------    
                    # Write data to CSV file 轉前csv
                    with open("D:/" + DIR_NAME + "/" + "Before_keypoints_data_" + str(predict_pose_number) + ".csv", mode="w", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerows(csv_data)
                        
                    
                    # Display the annotated frame
                    # cv2.imshow("windows_name", img)  
                    # cv2.imwrite("After_" + str(predict_pose_number) + ".jpg", color_image) # 轉後原圖
                    print("save done") 
                    print(f"CSV file 'keypoints_data_{str(predict_pose_number)}.csv' has been saved.")
                    #--------------------------------------------------------------------------  
                    
                    f_arm.arm_movej((radians(-90.07),
                                      radians(-70.8),
                                      radians(-78.95),             # 負的向下
                                      radians(-120.25),
                                      radians(90),
                                      radians(0.39)),acc,vel)  #校正角度
                    
                    time.sleep(1)
                                  
                    predict_pose_number += 1
                    '''
            if key & 0xFF == ord('p'):
                X, Y, W, H = 86, 41, 386, 436
                # roi = color_image[Y:Y+H, X:X+W]
                pose_model_name = "best_all_0_degree_small.v2i.v11l_pose.pt"
                seg_model_name = "best_Yat-sen_University_orchid-idea.v7i.v11s_seg.pt"
                # ALL_results_rows, img, csv_data, img_name, predict_time, angle_time = orchid_pose_predict_d435(color_image, depth_image, pose_model_name, predict_pose_number)
                # ALL_results_rows, img, csv_data, img_name, predict_time, angle_time, seg_time = orchid_pose_seg_area_leafs_number_predict_d435(color_image, depth_image, pose_model_name, seg_model_name, 
                #                                                                                                                                predict_pose_number)
                
                ALL_results_rows, img, csv_data, img_name, predict_time, angle_time, seg_time = orchid_pose_seg_area_leafs_number_predict_d435_new(color_image, depth_image, pose_model_name, seg_model_name, 
                                                                                                                                               predict_pose_number)
                
                if ALL_results_rows is None or len(ALL_results_rows) == 0:
                    print("No valid pose data detected.")
                    
                else:  
                    cv2.imwrite("D:/" + DIR_NAME + "/" + "Before_" + str(predict_pose_number) + ".jpg", color_image_copy) # 轉前原圖
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
                    cv2.imwrite("D:/" + DIR_NAME + "/" + img_name, img) # 轉前辨識圖
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
                        
                        # cv2.imwrite(img_name, img)
                        # print("save done") 
                        # print(f"CSV file 'keypoints_data_{str(predict_pose_number)}.csv' has been saved.")
                        
                        # time.sleep(5)
                        # print('wait')                
            
                        # print(B_temp)
                        
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
                            # posel = rob.getl()
                            # print(posel)
                            
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
                        
                    #--------------------------------------------------------------------------    
                    # Write data to CSV file 轉前csv
                    with open("D:/" + DIR_NAME + "/" + "Before_keypoints_data_" + str(predict_pose_number) + ".csv", mode="w", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerows(csv_data)
                        
                    
                    # Display the annotated frame
                    # cv2.imshow("windows_name", img)  
                    # cv2.imwrite("After_" + str(predict_pose_number) + ".jpg", color_image) # 轉後原圖
                    print("save done") 
                    print(f"CSV file 'keypoints_data_{str(predict_pose_number)}.csv' has been saved.")
                    #--------------------------------------------------------------------------  
                    
                    f_arm.arm_movej((radians(-90.07),
                                      radians(-70.8),
                                      radians(-78.95),             # 負的向下
                                      radians(-120.25),
                                      radians(90),
                                      radians(0.39)),acc,vel)  #校正角度
                    
                    time.sleep(1)
                                  
                    predict_pose_number += 1
                    #'''
            if key & 0xFF == ord('b'):
                #========================================================================== 轉後辨識  
                
                cv2.imwrite("D:/" + DIR_NAME + "/" + "After_" + str(predict_pose_number-1) + ".jpg", color_image_copy) # 轉後原圖
                
                pose_model_name = "best_all_0_degree_small.v2i.v11s-BiFPN_pose.pt"
                seg_model_name = "best_Yat-sen_University_orchid-idea.v7i.v11s_seg.pt"
                
                ALL_results_rows2, img2, csv_data2, img_name2, predict_time2, angle_time2, seg_time2 = orchid_pose_seg_area_leafs_number_predict_d435_new(color_image, depth_image, pose_model_name, seg_model_name, predict_pose_number)
                
                if ALL_results_rows2 is None or len(ALL_results_rows2) == 0:
                    print("No valid pose data detected.")
                    
                else:   
                    # Write data to CSV file 轉後csv
                    with open("D:/" + DIR_NAME + "/" + "After_keypoints_data_" + str(predict_pose_number-1) + ".csv", mode="w", newline="") as csvfile2:
                        csv_writer = csv.writer(csvfile2)
                        csv_writer.writerows(csv_data2)
                        
                    
                    # Display the annotated frame
                    # cv2.imshow("windows_name", img)
                    cv2.imwrite("D:/" + DIR_NAME + "/" + "After_predict" + str(predict_pose_number-1) + ".jpg", img2) # 轉後辯識圖
                    print("save done") 
                    print(f"CSV file 'keypoints_data_{str(predict_pose_number)}.csv' has been saved.")
                
                
                #==========================================================================

                
            if key == 27: #關閉視窗
                cv2.destroyAllWindows()
                break
finally:
    pipeline.stop()

 #leave some time to robot to process the setup commands

#rob.movel((pose[0],pose[1],pose[2],1.0372,2.5038,-2.5038),1,0.1)
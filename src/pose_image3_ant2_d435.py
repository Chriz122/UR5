import cv2
import csv
import math
import time
import random
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from algorithm.ant import create_distance_matrix, ant_colony_optimization
from orchid_pose_d435 import orchid_pose_predict_d435, orchid_pose_seg_area_predict_d435


if __name__ == "__main__":
    
    predict_pose_number = 1
    windows_name = "Camera Window"
    seg_model_name = "best_Yat-sen_University_orchid-idea.v6i.v8s-C2f-FasterBlock_seg3.pt"
    pose_model_name = "best_all_0_degree_small_keypoint.v7i.v8l-C2f-FasterBlock_pose_mpdiou_slideloss.pt"
    
    # D435i camera screen size
    Height_size = 720 ; Width_size = 1280
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置深度和颜色流
    #10、15或者30可选,20或者25会报错,其他帧率未尝试
    # 配置颜色相机
    config.enable_stream(rs.stream.color, Width_size, Height_size, rs.format.bgr8, 30)
    
    # 配置深度图像
    config.enable_stream(rs.stream.depth, Width_size, Height_size, rs.format.z16, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    start = time.time()
    
    # 创建对齐对象,rs.align 允许我们将深度帧与其他帧对齐,"align_to"是计划对其深度帧的流类型
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    while True:
        frames = pipeline.wait_for_frames()
            
        # 将深度框与颜色框对齐
        aligned_frames = align.process(frames)
        color_frame_p = aligned_frames.get_color_frame()
            
        #获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not aligned_depth_frame or not color_frame_p:
            continue
        color_frame = np.asanyarray(color_frame_p.get_data())
            
        # 将深度图转化为伪彩色图
        colorizer = rs.colorizer()
    
        # 對深度图黑洞區域進行填補
        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(aligned_depth_frame)
        depth_frame_modify = np.asanyarray(filled_depth.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    
        # 秀出RGB彩色图、深度伪彩色图
        images = np.hstack((color_frame, depth_colormap))
        cv2.namedWindow(windows_name, 0)
        cv2.imshow(windows_name, images)
        
        c = cv2.waitKey(1)
        
        if c & 0xFF == ord('\r') or c == 13 :
            ALL_results_rows, img, csv_data, img_name, predict_time, angle_time, segment_time = orchid_pose_seg_area_predict_d435(color_frame, depth_frame_modify, 
                                                                                                                         pose_model_name, seg_model_name, predict_pose_number)
            
            if ALL_results_rows is None or len(ALL_results_rows) == 0:
                print("No valid pose data detected.")
                
            else:  
                csv_data.append([])
                csv_data.append(["Predict-time(sec):", predict_time])
                csv_data.append(["Angle-time(sec):", angle_time])
                csv_data.append(["Segment-time(sec):", segment_time])
                # print(ALL_results_rows)
                # 创建距离矩阵
                distance_matrix = create_distance_matrix(ALL_results_rows)
                
                # 使用蚁群算法规划路线
                best_route_indices, best_distance, ant_time = ant_colony_optimization(distance_matrix)
                csv_data.append(["Ant Path-execute-time(sec):", ant_time])
                
                best_route = [ALL_results_rows[i] for i in best_route_indices]
                
                # 打印总距离
                print(f"总距离: {best_distance}")
                
                # 绘制路径
                for i in range(len(best_route)):
                    loc = best_route[i]
                    
                    if i > 0:
                        prev_loc = best_route[i-1]
                        cv2.line(img, tuple(prev_loc[1]), tuple(loc[1]), (255, 0, 255), 2)
                    
                        if loc[2][0] < 0:
                            # 夾爪順時針轉
                            print(f'夾爪順時針轉{abs(loc[2][0])}度') # 夾爪要對齊植株介質的角度
                            
                            if loc[3] < 0:   
                                print(f'夾爪逆時針轉{abs(loc[3])}度') # 夾爪要轉動植株的角度        
                            else:
                                print(f'夾爪順時針轉{abs(loc[3])}度') # 夾爪要轉動植株的角度
                            
                        elif loc[2][0] > 0: 
                            # 夾爪逆時針轉
                            print(f'夾爪逆時針轉{abs(loc[2][0])}度') # 夾爪要對齊植株介質的角度
                            
                            if loc[3] < 0:   
                                print(f'夾爪逆時針轉{abs(loc[3])}度') # 夾爪要轉動植株的角度        
                            else:
                                print(f'夾爪順時針轉{abs(loc[3])}度') # 夾爪要轉動植株的角度          
                    
                    
                # Write data to CSV file
                with open("keypoints_data_" + str(predict_pose_number) + ".csv", mode="w", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerows(csv_data)
                    
                
                # Display the annotated frame
                # cv2.imshow(windows_name, img)
                cv2.imwrite(img_name, img)
                print("save done") 
                print(f"CSV file 'keypoints_data_{str(predict_pose_number)}.csv' has been saved.")
                predict_pose_number += 1
            
        
        # 如果按下(ESC)则关闭窗口(ESC的ascii码为27),同时跳出循环
        if c & 0xFF == ord('\x1b') or c == 27:
            cv2.destroyAllWindows()
            pipeline.stop()
            break

    
    
    
    
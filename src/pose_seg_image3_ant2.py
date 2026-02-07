import cv2
import csv
import math
import random
import numpy as np
from ultralytics import YOLO

from algorithm.ant import create_distance_matrix, ant_colony_optimization
from orchid_pose import orchid_pose_predict, orchid_pose_seg_area_predict, orchid_pose_seg_area_leafs_number_predict


if __name__ == "__main__":
    
    windows_name = "Camera Window"
    pose_model_name = "best_all_0_degree_small_keypoint.v7i.v8l-C2f-FasterBlock_pose_mpdiou_slideloss.pt"
    seg_model_name = "best_Yat-sen_University_orchid-idea.v6i.v8s-C2f-FasterBlock_seg3.pt"

    # img = cv2.imread(r"D:\ultralytics-main\pose\0ds-1872-_bmp.rf.026efcc96422d88ab9255ea5b1bf1eed.jpg")
    img = cv2.imread(r"D:\ultralytics-main\pose\工業電腦\o2.jpg")
    
    predict_pose_number = 1
    # ALL_results_rows, img, csv_data, img_name, predict_time, angle_time = orchid_pose_predict(img, pose_model_name, predict_pose_number)
    ALL_results_rows, img, csv_data, img_name, exetime, exetime2, exetime3 = orchid_pose_seg_area_predict(img, pose_model_name, 
                                                                                                          seg_model_name, predict_pose_number)
    
    if ALL_results_rows is None or len(ALL_results_rows) == 0:
        print("No valid pose data detected.")
        cv2.imwrite(img_name, img)
        print("save done") 
        
    else: 
        # print(ALL_results_rows)
        csv_data.append([])
        csv_data.append(["Predict-time(sec):", exetime])
        csv_data.append(["Angle-time(sec):", exetime2])
        csv_data.append(["Segment-time(sec):", exetime3])
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
            
                if loc[2] < 0:
                    # 夾爪順時針轉
                    print(f'夾爪順時針轉{abs(loc[2])}度') # 夾爪要對齊植株介質的角度
                    
                    if loc[3] < 0:   
                        print(f'夾爪逆時針轉{abs(loc[3])}度') # 夾爪要轉動植株的角度        
                    else:
                        print(f'夾爪順時針轉{abs(loc[3])}度') # 夾爪要轉動植株的角度
                    
                elif loc[2] > 0: 
                    # 夾爪逆時針轉
                    print(f'夾爪逆時針轉{abs(loc[2])}度') # 夾爪要對齊植株介質的角度
                    
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
    
    
    
    
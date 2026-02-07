import torch
import time
import math
import cv2
import numpy as np
from hex2rgb import hex2rgb
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from orchid_seg import orchid_seg_predict, orchid_seg_predict_block

if __name__ == "__main__":
    
    windows_name = "Camera Window"
    seg_model_name = "best_Yat-sen_University_orchid-idea.v6i.v8s-C2f-FasterBlock_seg3.onnx"

    img = cv2.imread(r"D:\ultralytics-main\pose\0ds-1872-_bmp.rf.026efcc96422d88ab9255ea5b1bf1eed.jpg")
    
    predict_pose_number = 1
    # ALL_results_rows, img, csv_data, img_name, predict_time, angle_time = orchid_pose_predict(img, pose_model_name, predict_pose_number)
    img, img_name = orchid_seg_predict(img, seg_model_name, predict_pose_number)  
    
    # Display the annotated frame
    # cv2.imshow(windows_name, img)
    cv2.imwrite(img_name, img)
    print("save done") 
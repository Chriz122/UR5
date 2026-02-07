import time
import cv2
import csv
import math
import random
import numpy as np
from ultralytics import YOLO

from angle import *
from orchid_seg import orchid_seg_predict_block, orchid_seg_leafs_number_predict_block, orchid_seg_leafs_number_predict_block2

def distance(point1, point2):
    # 計算兩點之間的歐幾里得距離
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def check_point_to_points(point1, points):
    # 檢查point1與points中嵌套的每個點之間的距離
    for point_info in points:
        point = point_info[0]  # 獲取每個子列表中的點
        if distance(point1, point) <= 50:
            return [point_info[1], point_info[2]]  # 如果距離不超過10，返回1
    return [0, 0]  # 否則，返回0


def orchid_RGB_modified(img, dot):
    # 定義檢查範圍直徑
    diameter = 8

    # 圖像大小
    height, width, _ = img.shape

    # 隨機點座標
    x, y = dot[0], dot[1]

    # 遍歷以(x, y)為中心，直徑為10的範圍
    for i in range(-(diameter // 2), (diameter // 2) + 1):
        for j in range(-(diameter // 2), (diameter // 2) + 1):
            # 計算當前檢查點的座標
            check_x = x + i
            check_y = y + j

            # 確保座標在圖像範圍內
            if 0 <= check_x < width and 0 <= check_y < height:
                # 獲取當前點的顏色
                (b, g, r) = img[check_y, check_x]
                # print((b, g, r))
                # 判斷顏色，並返回對應的值
                if (b, g, r) == (56, 56, 255):  # 紅色
                    # print("該點及其直徑10範圍內存在紅色區域。")
                    return 5
                elif (b, g, r) == (151, 157, 255):  # 綠色
                    # print("該點及其直徑10範圍內存在綠色區域。")
                    return 4
                elif (b, g, r) == (29, 178, 255):  # 藍色
                    # print("該點及其直徑10範圍內存在藍色區域。")
                    return 3
                elif (b, g, r) == (49, 210, 207):  # 藍色
                    # print("該點及其直徑10範圍內存在藍色區域。")
                    return 2

    # 如果範圍內沒有指定的顏色，返回0
    # print("該點及其直徑10範圍內沒有紅色、綠色或藍色區域。")
    return 0


def orchid_RGB(img, dot):
    # 定義檢查範圍直徑
    diameter = 3

    # 圖像大小
    height, width, _ = img.shape

    # 隨機點座標
    x, y = dot[0], dot[1]
    # print(x, y)

    # 設定標誌變數
    is_in_black_area = True

    # 遍歷以(x, y)為中心，直徑為10的範圍
    for i in range(-(diameter // 2), (diameter // 2) + 1):
        for j in range(-(diameter // 2), (diameter // 2) + 1):
            # 計算當前檢查點的座標
            check_x = x + i
            check_y = y + j
            # print(check_x, check_y)
            # 確保座標在圖像範圍內
            if 0 <= check_x < width and 0 <= check_y < height:
                # 如果有任何一個點不是黑色，則標誌為False
                (b, g, r) = img[check_y, check_x]
                # print((b, g, r))
                if b > 0 or g > 0 or r > 0:  # 用RGB來判斷黑色
                    is_in_black_area = False
                    break
        if not is_in_black_area:
            break

    # 判斷結果
    if is_in_black_area:
        # print("該點及其直徑10範圍內都在黑色區塊內。")
        return True
    else:
        # print("該點及其直徑10範圍內不全在黑色區塊內。")
        return False


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


def orchid_pose_predict(img, pose_model_name, predict_pose_number):
    
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    skeleton = [[2, 1], [3, 1], [4, 1], [5, 1]]

    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                             [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                             [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                             [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)

    # kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    kpt_color  = pose_palette[[10, 0, 9, 7, 16]]

    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    ALL_results_rows = []
    
    #---------------------------------------------------------------  
    
    model = YOLO(pose_model_name, task='pose')
    img_name = "predict-pose-single" + str(predict_pose_number) + ".jpg"
    
    start_time = time.time()
        
    results = model.track(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False, tracker = "bytetrack.yaml", persist = True)[0]
    
    end_time = time.time()
    exetime = end_time - start_time # 辨識時間
    
    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype = "int")

    # keypoints.data.shape -> n,17,3
    keypoints = results.keypoints.cpu().numpy()
    # Prepare CSV data
    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]
    
    start_time_2 = time.time()
    # keypoint -> 每个人的关键点
    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        # print(keypoint)
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        centerx= int((left + right)/2); centery= int((top + bottom)/2);
        
        row = [id]; nrow = [id]; results_rows = [id]; # 把結果放在【results_rows】裡面
        for i, (x, y, conf) in enumerate(keypoint):
            color_k = [int(x) for x in kpt_color[i]]
            row.append(f"({int(x)}, {int(y)})")
            nrow.append([int(x), int(y)])
            cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)
            
            # color_k = [int(x) for x in random_color(i)]
            # if conf < 0.5:
            #     row.append(f"({int(x)}, {int(y)})")
            #     # continue
            # if x != 0 and y != 0:
            #     row.append(f"({int(x)}, {int(y)})")
            #     cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)  
            
            # if i == 0 or i == 1 or i == 2:
            #     results_rows.append([int(x), int(y)])
            if i == 0:
                results_rows.append([int(x), int(y)])

        angle_soil = int(calculate_angle_soil(left, top, right, bottom, nrow[2][0], nrow[2][1], 
                        nrow[3][0], nrow[3][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom, nrow[4][0], nrow[4][1], 
                        nrow[5][0], nrow[5][1])) % 180
        
        row.append(angle_soil)
        
        if angle_rotate > 90:
            angle_rotate = angle_rotate -180  
        row.append(angle_rotate) 
        
        # csv_data.append(row) 
        
        if angle_rotate != 0 and (abs(angle_rotate) >= 10):
            results_rows.append(int(angle_soil))
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            # Draw angle
            x0, y0 = nrow[1]
            x1, y1 = nrow[2]
            x2, y2 = nrow[3]
            x3, y3 = nrow[4]
            x4, y4 = nrow[5]
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
    
            # Line from (x3, y3) to (x4, y4)
            cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)
            
            # Line from (x3, y3) to (x4, y4)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
            # Vertical line through the center of the bounding box
            cv2.line(img, (left, y0), (right, y0), (0, 0, 255), 2)
    
            # Draw the angle text
            cv2.putText(img, f'{int(angle_soil)} {int(angle_rotate)} deg', (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        else:  
            pass
        
        csv_data.append(row) 
        
        for i, sk in enumerate(skeleton):
            pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
            pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
            
            # print(pos1, pos2)

            conf1 = keypoint[(sk[0] - 1), 2]
            conf2 = keypoint[(sk[1] - 1), 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                continue
            
            cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)


        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        centerx= int((left + right)/2); centery= int((top + bottom)/2);
        # cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
        
        # caption = f"{id} {names[label]} {confidence:.2f}"
        # caption = f"{int(angle)}"
        caption = f"{id}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]     
        # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
        # cv2.putText(img, caption, (left, top - 5), 0, 1, (255, 255, 255), 2, 16)
        # cv2.rectangle(img, (centerx - 3, centery - 33), (centerx + w + 10, centery), color, -1)
        # cv2.putText(img, caption, (centerx, centery - 5), 0, 1, (255, 255, 255), 2, 16)
    
    end_time_2 = time.time()
    exetime2 = end_time_2 - start_time_2 # 辨識時間
    return ALL_results_rows, img, csv_data, img_name, exetime, exetime2


def orchid_pose_seg_area_predict(img, pose_model_name, seg_model_name, predict_pose_number):
    
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    skeleton = [[2, 1], [3, 1], [4, 1], [5, 1]]

    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                             [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                             [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                             [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)

    # kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    kpt_color  = pose_palette[[10, 0, 9, 7, 4]]

    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    ALL_results_rows = []
    
    #---------------------------------------------------------------  
    
    model = YOLO(pose_model_name, task='pose')
    img_name = "predict-pose-seg-single" + str(predict_pose_number) + ".jpg"
    
    start_time = time.time()
        
    results = model.track(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False, tracker = "bytetrack.yaml", persist = True)[0]
    
    end_time = time.time()
    exetime = end_time - start_time # 辨識時間
    
    #--------------------------------------------------------------- 
    
    #================================================================================   
    
    start_time_3 = time.time()
    
    # seg_model_name = "best_Yat-sen_University_orchid-idea.v6i.v8s-SENetV2_seg.pt"
    img_seg, img_seg_name = orchid_seg_predict_block(img, seg_model_name, predict_pose_number) # 分割
    
    end_time_3 = time.time()
    exetime3 = end_time_3 - start_time_3 # 辨識時間
    
    img_seg_copy = img_seg.copy()
    
    #================================================================================ 
    
    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype = "int")

    # keypoints.data.shape -> n,17,3
    keypoints = results.keypoints.cpu().numpy()
    # Prepare CSV data
    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]
    
    start_time_2 = time.time()
    # keypoint -> 每个人的关键点
    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        # print(keypoint)
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        centerx= int((left + right)/2); centery= int((top + bottom)/2);
        
        row = [id]; nrow = [id]; results_rows = [id];
        for i, (x, y, conf) in enumerate(keypoint):
            color_k = [int(x) for x in kpt_color[i]]
            row.append(f"({int(x)}, {int(y)})")
            nrow.append([int(x), int(y)])
            cv2.circle(img_seg, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)
            
            # color_k = [int(x) for x in random_color(i)]
            # if conf < 0.5:
            #     row.append(f"({int(x)}, {int(y)})")
            #     # continue
            # if x != 0 and y != 0:
            #     row.append(f"({int(x)}, {int(y)})")
            #     cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)  
            
            # if i == 0 or i == 1 or i == 2:
            #     results_rows.append([int(x), int(y)])
            if i == 0:
                results_rows.append([int(x), int(y)])
                
        if orchid_RGB(img_seg_copy, [nrow[2][0], nrow[2][1]]) and orchid_RGB(img_seg_copy, [nrow[3][0], nrow[3][1]]):

            angle_soil = int(calculate_angle_soil(left, top, right, bottom, nrow[2][0], nrow[2][1], 
                            nrow[3][0], nrow[3][1]))
            angle_rotate = int(calculate_angle_rotate(left, top, right, bottom, nrow[4][0], nrow[4][1], 
                            nrow[5][0], nrow[5][1])) % 180
            
            row.append(angle_soil)
            
            if angle_rotate > 90:
                angle_rotate = angle_rotate -180  
            row.append(angle_rotate) 
            
            # csv_data.append(row) 
            
            if angle_rotate != 0 and (abs(angle_rotate) >= 10):
                results_rows.append(int(angle_soil))
                results_rows.append(int(angle_rotate))
                ALL_results_rows.append(results_rows)
                # Draw angle
                x0, y0 = nrow[1]
                x1, y1 = nrow[2]
                x2, y2 = nrow[3]
                x3, y3 = nrow[4]
                x4, y4 = nrow[5]
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
        
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)
                
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
                # Vertical line through the center of the bounding box
                cv2.line(img, (left, y0), (right, y0), (0, 0, 255), 2)
        
                # Draw the angle text
                cv2.putText(img, f'{int(angle_soil)} {int(angle_rotate)} deg', (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            else:  
                pass
            
            csv_data.append(row) 
            
            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
                pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
                
                # print(pos1, pos2)
    
                conf1 = keypoint[(sk[0] - 1), 2]
                conf2 = keypoint[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
    
    
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            centerx= int((left + right)/2); centery= int((top + bottom)/2);
            # cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
            
            # caption = f"{id} {names[label]} {confidence:.2f}"
            # caption = f"{int(angle)}"
            caption = f"{id}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]     
            # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            # cv2.putText(img, caption, (left, top - 5), 0, 1, (255, 255, 255), 2, 16)
            # cv2.rectangle(img, (centerx - 3, centery - 33), (centerx + w + 10, centery), color, -1)
            # cv2.putText(img, caption, (centerx, centery - 5), 0, 1, (255, 255, 255), 2, 16)
    
    end_time_2 = time.time()
    exetime2 = end_time_2 - start_time_2 # 辨識時間
    return ALL_results_rows, img_seg, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_seg_area_leafs_number_predict(img, pose_model_name, seg_model_name, predict_pose_number):
    
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    skeleton = [[2, 1], [3, 1], [4, 1], [5, 1]]

    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                             [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                             [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                             [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)

    # kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    kpt_color  = pose_palette[[10, 0, 9, 7, 16]]

    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    ALL_results_rows = []
    
    #---------------------------------------------------------------  
    
    model = YOLO(pose_model_name, task='pose')
    img_name = "predict-pose-seg-single" + str(predict_pose_number) + ".jpg"
    
    start_time = time.time()
        
    results = model.track(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False, tracker = "bytetrack.yaml", persist = True)[0]
    
    end_time = time.time()
    exetime = end_time - start_time # 辨識時間
    
    #--------------------------------------------------------------- 
    
    #================================================================================   
    
    start_time_3 = time.time()
    
    # seg_model_name = "best_Yat-sen_University_orchid-idea.v6i.v8s-SENetV2_seg.pt"
    img_seg, img_seg_name, results_row_leafs_seg = orchid_seg_leafs_number_predict_block2(img, seg_model_name, predict_pose_number) # 分割

    end_time_3 = time.time()
    exetime3 = end_time_3 - start_time_3 # 辨識時間
    
    # print(results_row_leafs_seg)
    img_seg_copy = img_seg.copy()
    
    #================================================================================ 
    
    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype = "int")

    # keypoints.data.shape -> n,17,3
    keypoints = results.keypoints.cpu().numpy()
    # Prepare CSV data
    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Leafs-Number", "Soil-Angle", "Rotate-Angle"]]
    
    start_time_2 = time.time()
    # keypoint -> 每个人的关键点
    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        # print(keypoint)
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        centerx= int((left + right)/2); centery= int((top + bottom)/2);
        
        row = [id]; nrow = [id]; results_rows = [id];
        for i, (x, y, conf) in enumerate(keypoint):
            color_k = [int(x) for x in kpt_color[i]]
            row.append(f"({int(x)}, {int(y)})")
            nrow.append([int(x), int(y)])
            cv2.circle(img_seg, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)
            
            # color_k = [int(x) for x in random_color(i)]
            # if conf < 0.5:
            #     row.append(f"({int(x)}, {int(y)})")
            #     # continue
            # if x != 0 and y != 0:
            #     row.append(f"({int(x)}, {int(y)})")
            #     cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)  
            
            # if i == 0 or i == 1 or i == 2:
            #     results_rows.append([int(x), int(y)])
            if i == 0:
                results_rows.append([int(x), int(y)])
                
                
        if orchid_RGB(img_seg_copy, [nrow[2][0], nrow[2][1]]) and orchid_RGB(img_seg_copy, [nrow[3][0], nrow[3][1]]):
            
            # leafs_num = orchid_RGB_modified(img_seg_copy, [nrow[1][0], nrow[1][1]]) # 葉子數
            # results_rows.append(leafs_num)
            # row.append(leafs_num)
            
            needed_leafs = check_point_to_points([nrow[1][0], nrow[1][1]], results_row_leafs_seg)
            # print(needed_leafs)
            results_rows.append(needed_leafs)
            row.append(needed_leafs[1])
            
            angle_soil = int(calculate_angle_soil(left, top, right, bottom, nrow[2][0], nrow[2][1], 
                            nrow[3][0], nrow[3][1]))
            angle_rotate = int(calculate_angle_rotate(left, top, right, bottom, nrow[4][0], nrow[4][1], 
                            nrow[5][0], nrow[5][1])) % 180
            
            row.append(angle_soil)
            
            if angle_rotate > 90:
                angle_rotate = angle_rotate -180  
            row.append(angle_rotate) 
            
            # csv_data.append(row) 
            
            if angle_rotate != 0 and (abs(angle_rotate) >= 10):
                results_rows.append(int(angle_soil))
                results_rows.append(int(angle_rotate))
                ALL_results_rows.append(results_rows)
                # Draw angle
                x0, y0 = nrow[1]
                x1, y1 = nrow[2]
                x2, y2 = nrow[3]
                x3, y3 = nrow[4]
                x4, y4 = nrow[5]
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
        
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)
                
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
                # Vertical line through the center of the bounding box
                cv2.line(img, (left, y0), (right, y0), (0, 0, 255), 2)
        
                # Draw the angle text
                cv2.putText(img, f'{int(angle_soil)} {int(angle_rotate)} deg', (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            else:  
                pass
            
            csv_data.append(row) 
            
            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
                pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
                
                # print(pos1, pos2)
    
                conf1 = keypoint[(sk[0] - 1), 2]
                conf2 = keypoint[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
    
    
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            centerx= int((left + right)/2); centery= int((top + bottom)/2);
            # cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
            
            # caption = f"{id} {names[label]} {confidence:.2f}"
            # caption = f"{int(angle)}"
            caption = f"{id}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]     
            # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            # cv2.putText(img, caption, (left, top - 5), 0, 1, (255, 255, 255), 2, 16)
            # cv2.rectangle(img, (centerx - 3, centery - 33), (centerx + w + 10, centery), color, -1)
            # cv2.putText(img, caption, (centerx, centery - 5), 0, 1, (255, 255, 255), 2, 16)
    
    end_time_2 = time.time()
    exetime2 = end_time_2 - start_time_2 # 辨識時間
    return ALL_results_rows, img_seg, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_predict_d435(img, depth_frame, pose_model_name, predict_pose_number):
    
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    skeleton = [[2, 1], [3, 1], [4, 1], [5, 1]]

    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                             [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                             [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                             [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)

    # kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    kpt_color  = pose_palette[[10, 0, 9, 7, 16]]

    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    ALL_results_rows = []
    
    #---------------------------------------------------------------  
    
    model = YOLO(pose_model_name)
    img_name = "predict-pose-single" + str(predict_pose_number) + ".jpg"
    
    start_time = time.time()
        
    results = model.track(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False, tracker = "bytetrack.yaml", persist = True)[0]
    
    end_time = time.time()
    exetime = end_time - start_time # 辨識時間
    
    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0  # Return tuple with default values
    
    else:
        names = results.names
        boxes = results.boxes.data.tolist()
        ids = np.array(results.boxes.id.cpu(), dtype = "int")
    
        # keypoints.data.shape -> n,17,3
        keypoints = results.keypoints.cpu().numpy()
        # Prepare CSV data
        csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]
        
        start_time_2 = time.time()
        # keypoint -> 每个人的关键点
        for obj, keypoint, id in zip(boxes, keypoints.data, ids):
            # print(keypoint)
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            centerx= int((left + right)/2); centery= int((top + bottom)/2);
            
            row = [id]; nrow = [id]; results_rows = [id];
            for i, (x, y, conf) in enumerate(keypoint):
                color_k = [int(x) for x in kpt_color[i]]
                row.append(f"({int(x)}, {int(y)})")
                nrow.append([int(x), int(y)])
                cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)
                
                # color_k = [int(x) for x in random_color(i)]
                # if conf < 0.5:
                #     row.append(f"({int(x)}, {int(y)})")
                #     # continue
                # if x != 0 and y != 0:
                #     row.append(f"({int(x)}, {int(y)})")
                #     cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)  
                
                # if i == 0 or i == 1 or i == 2:
                #     results_rows.append([int(x), int(y)])
                if i == 0:
                    results_rows.append([int(x), int(y)])
                    
            
            angle_soil = int(calculate_angle_soil(left, top, right, bottom, nrow[2][0], nrow[2][1], 
                            nrow[3][0], nrow[3][1]))
            angle_rotate = int(calculate_angle_rotate(left, top, right, bottom, nrow[4][0], nrow[4][1], 
                            nrow[5][0], nrow[5][1])) % 180
            
            soil_centroid_x = int((nrow[2][0]+nrow[3][0])/2); soil_centroid_y = int((nrow[2][1]+nrow[3][1])/2);
            soil_distance = math.sqrt( ((nrow[2][0]-nrow[3][0])**2) + 
                                      ((nrow[2][1]-nrow[3][1])**2) + 
                                      ((float(depth_frame[nrow[2][1]][nrow[2][0]])-float(depth_frame[nrow[3][1]][nrow[3][0]]))**2) )        
            row.append(angle_soil)
            
            results_rows.append([angle_soil, [soil_centroid_x, soil_centroid_y], soil_distance])
            
            if angle_rotate > 90:
                angle_rotate = angle_rotate -180  
            row.append(angle_rotate) 
            
            # csv_data.append(row) 
            
            if angle_rotate != 0 and (abs(angle_rotate) >= 10):
                results_rows.append(int(angle_rotate))
                ALL_results_rows.append(results_rows)
                # Draw angle
                x0, y0 = nrow[1]
                x1, y1 = nrow[2]
                x2, y2 = nrow[3]
                x3, y3 = nrow[4]
                x4, y4 = nrow[5]
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
        
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)
                
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
                # Vertical line through the center of the bounding box
                cv2.line(img, (left, y0), (right, y0), (0, 0, 255), 2)
        
                # Draw the angle text
                cv2.putText(img, f'{int(angle_soil)} {int(angle_rotate)} deg', (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            else:  
                pass
            
            csv_data.append(row) 
            
            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
                pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
                
                # print(pos1, pos2)
    
                conf1 = keypoint[(sk[0] - 1), 2]
                conf2 = keypoint[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
    
    
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            centerx= int((left + right)/2); centery= int((top + bottom)/2);
            # cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
            
            # caption = f"{id} {names[label]} {confidence:.2f}"
            # caption = f"{int(angle)}"
            caption = f"{id}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]     
            # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            # cv2.putText(img, caption, (left, top - 5), 0, 1, (255, 255, 255), 2, 16)
            # cv2.rectangle(img, (centerx - 3, centery - 33), (centerx + w + 10, centery), color, -1)
            # cv2.putText(img, caption, (centerx, centery - 5), 0, 1, (255, 255, 255), 2, 16)
        
        end_time_2 = time.time()
        exetime2 = end_time_2 - start_time_2 # 辨識時間
        
        return ALL_results_rows, img, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_predict_d435_new(img, depth_frame, pose_model_name, predict_pose_number):
    
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    skeleton = [[2, 1], [3, 1], [4, 1], [5, 1]]

    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                             [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                             [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                             [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)

    # kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    kpt_color  = pose_palette[[10, 0, 9, 7, 16]]

    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    ALL_results_rows = []
    
    #---------------------------------------------------------------  
    
    model = YOLO(pose_model_name)
    img_name = "predict-pose-single" + str(predict_pose_number) + ".jpg"
    
    start_time = time.time()
        
    results = model.track(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False, tracker = "bytetrack.yaml", persist = True)[0]
    
    end_time = time.time()
    exetime = end_time - start_time # 辨識時間
    
    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0  # Return tuple with default values
    
    else:
        names = results.names
        boxes = results.boxes.data.tolist()
        ids = np.array(results.boxes.id.cpu(), dtype = "int")
    
        # keypoints.data.shape -> n,17,3
        keypoints = results.keypoints.cpu().numpy()
        # Prepare CSV data
        csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]
        
        start_time_2 = time.time()
        # keypoint -> 每个人的关键点
        for obj, keypoint, id in zip(boxes, keypoints.data, ids):
            # print(keypoint)
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            centerx= int((left + right)/2); centery= int((top + bottom)/2);
            
            row = [id]; nrow = [id]; results_rows = [id];
            for i, (x, y, conf) in enumerate(keypoint):
                color_k = [int(x) for x in kpt_color[i]]
                row.append(f"({int(x)}, {int(y)})")
                nrow.append([int(x), int(y)])
                cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)
                
                # color_k = [int(x) for x in random_color(i)]
                # if conf < 0.5:
                #     row.append(f"({int(x)}, {int(y)})")
                #     # continue
                # if x != 0 and y != 0:
                #     row.append(f"({int(x)}, {int(y)})")
                #     cv2.circle(img, (int(x), int(y)), 5, color_k , -1, lineType=cv2.LINE_AA)  
                
                # if i == 0 or i == 1 or i == 2:
                #     results_rows.append([int(x), int(y)])
                if i == 0:
                    results_rows.append([int(x), int(y)])
    
            angle_soil = int(calculate_angle_soil(left, top, right, bottom, nrow[4][0], nrow[4][1], 
                            nrow[5][0], nrow[5][1]))
            angle_rotate = int(calculate_angle_rotate(left, top, right, bottom, nrow[2][0], nrow[2][1], 
                            nrow[3][0], nrow[3][1])) % 180
            
            soil_centroid_x = int((nrow[4][0]+nrow[5][0])/2); soil_centroid_y = int((nrow[4][1]+nrow[5][1])/2);
            soil_distance = math.sqrt( ((nrow[4][0]-nrow[5][0])**2) + 
                                      ((nrow[4][1]-nrow[5][1])**2) + 
                                      ((float(depth_frame[nrow[4][1]][nrow[4][0]])-float(depth_frame[nrow[5][1]][nrow[5][0]]))**2) )
            row.append(angle_soil)
            
            results_rows.append([angle_soil, [soil_centroid_x, soil_centroid_y], soil_distance])
            
            if angle_rotate > 90:
                angle_rotate = angle_rotate -180  
            row.append(angle_rotate) 
            
            # csv_data.append(row) 
            
            if angle_rotate != 0 and (abs(angle_rotate) >= 10):
                results_rows.append(int(angle_rotate))
                ALL_results_rows.append(results_rows)
                # Draw angle
                x0, y0 = nrow[1]
                x1, y1 = nrow[2]
                x2, y2 = nrow[3]
                x3, y3 = nrow[4]
                x4, y4 = nrow[5]
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
        
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)
                
                # Line from (x3, y3) to (x4, y4)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
                # Vertical line through the center of the bounding box
                cv2.line(img, (left, y0), (right, y0), (0, 0, 255), 2)
        
                # Draw the angle text
                cv2.putText(img, f'{int(angle_soil)} {int(angle_rotate)} deg', (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            else:  
                pass
            
            csv_data.append(row) 
            
            for i, sk in enumerate(skeleton):
                pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1), 1]))
                pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1), 1]))
                
                # print(pos1, pos2)
    
                conf1 = keypoint[(sk[0] - 1), 2]
                conf2 = keypoint[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
    
    
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            centerx= int((left + right)/2); centery= int((top + bottom)/2);
            # cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
            
            # caption = f"{id} {names[label]} {confidence:.2f}"
            # caption = f"{int(angle)}"
            caption = f"{id}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]     
            # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            # cv2.putText(img, caption, (left, top - 5), 0, 1, (255, 255, 255), 2, 16)
            # cv2.rectangle(img, (centerx - 3, centery - 33), (centerx + w + 10, centery), color, -1)
            # cv2.putText(img, caption, (centerx, centery - 5), 0, 1, (255, 255, 255), 2, 16)
        
        end_time_2 = time.time()
        exetime2 = end_time_2 - start_time_2 # 辨識時間
        return ALL_results_rows, img, csv_data, img_name, exetime, exetime2
        
        
        
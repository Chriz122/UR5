import os
import cv2
import csv
import math
import time
import random
import numpy as np
from ultralytics import YOLO

# 计算两点间距离的函数
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 初始化距离矩阵
def create_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = distance(locations[i][1], locations[j][1])
    return distance_matrix

# 蚁群算法
def ant_colony_optimization(distance_matrix, n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    n = len(distance_matrix)
    pheromones = np.ones((n, n)) / n
    best_route = None
    best_distance = float('inf')

    for _ in range(n_iterations):
        all_routes = []
        all_distances = []
        
        for _ in range(n_ants):
            route = []
            visited = set()
            current_city = random.randint(0, n - 1)
            route.append(current_city)
            visited.add(current_city)
            
            for _ in range(n - 1):
                probabilities = []
                for next_city in range(n):
                    if next_city not in visited:
                        pheromone = pheromones[current_city][next_city] ** alpha
                        heuristic = (1.0 / distance_matrix[current_city][next_city]) ** beta
                        probabilities.append((pheromone * heuristic, next_city))
                
                total_prob = sum([prob[0] for prob in probabilities])
                probabilities = [(prob[0] / total_prob, prob[1]) for prob in probabilities]
                probabilities.sort(reverse=True)
                
                rand_prob = random.random()
                cumulative_prob = 0.0
                for prob, next_city in probabilities:
                    cumulative_prob += prob
                    if rand_prob <= cumulative_prob:
                        route.append(next_city)
                        visited.add(next_city)
                        current_city = next_city
                        break

            all_routes.append(route)
            total_distance = sum([distance_matrix[route[i - 1]][route[i]] for i in range(n)])
            all_distances.append(total_distance)

            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1.0 - evaporation_rate)

        for route, total_distance in zip(all_routes, all_distances):
            for i in range(n):
                pheromones[route[i - 1]][route[i]] += 1.0 / total_distance

    return best_route, best_distance

def calculate_angle_soil(left, top, right, bottom, x3, y3, x4, y4):
    # 如果其中一個點為(0, 0)，則不計算角度
    if (x3 == 0 and y3 == 0) or (x4 == 0 and y4 == 0):
        # print("One of the points is (0, 0), angle calculation skipped.")
        angle_in_degrees = 0.0
        
    else:
        # Step 1: Calculate the center x-coordinate of the rectangle
        center_x = (left + right) / 2
    
        # Step 2: Compute the angle of the blue line with respect to the horizontal axis
        delta_x = x4 - x3
        delta_y = y4 - y3
        angle_blue_line = math.atan2(delta_y, delta_x)
    
        # Step 3: Compute the angle between the red vertical line and the blue line
        # Since the red line is vertical, its angle with the horizontal is 90 degrees (π/2 radians)
        angle_vertical_line = math.pi / 2
    
        # Calculate the angle between the vertical red line and the blue line
        angle_between_lines = angle_vertical_line - angle_blue_line
    
        # Convert the angle to degrees
        angle_in_degrees = math.degrees(angle_between_lines)
        
    return angle_in_degrees

def calculate_angle_rotate(left, top, right, bottom, x3, y3, x4, y4):
    # 如果其中一個點為(0, 0)，則不計算角度
    if (x3 == 0 and y3 == 0) or (x4 == 0 and y4 == 0):
        # print("One of the points is (0, 0), angle calculation skipped.")
        angle_in_degrees = 0.0
        
    else:
        # Step 1: Calculate the center x-coordinate of the rectangle
        center_x = (top + bottom) / 2
    
        # Step 2: Compute the angle of the blue line with respect to the horizontal axis
        delta_x = x4 - x3
        delta_y = y4 - y3
        angle_blue_line = math.atan2(delta_y, delta_x)
    
        # Step 3: Compute the angle between the red vertical line and the blue line
        # Since the red line is vertical, its angle with the horizontal is 90 degrees (π/2 radians)
        angle_vertical_line = math.pi
    
        # Calculate the angle between the vertical red line and the blue line
        angle_between_lines = angle_vertical_line - angle_blue_line
    
        # Convert the angle to degrees
        angle_in_degrees = math.degrees(angle_between_lines)
        
    return angle_in_degrees

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

def print_all_file_paths(directory):
    model = YOLO("best_all_0_degree_small_keypoint.v7i.v8l-C2f-FasterBlock_pose_mpdiou_slideloss.pt")
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        I = 0
        for file in files:
            ALL_results_rows = []
            # Print the full file path
            img = cv2.imread(os.path.join(root, file))
            img_name = "predict-pose-" + file
                
            results = model.track(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                                    save = False, tracker = "bytetrack.yaml", persist = True)[0]
            
            names = results.names
            boxes = results.boxes.data.tolist()
            ids = np.array(results.boxes.id.cpu(), dtype = "int")

            # keypoints.data.shape -> n,17,3
            keypoints = results.keypoints.cpu().numpy()
            # Prepare CSV data
            csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)"]]
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
                
                # soil_centroid_x =  int( (nrow[2][0]+nrow[3][0])/2 ); soil_centroid_y =  int( (nrow[2][1]+nrow[3][1])/2 );
                
                # soil_centroid = [soil_centroid_x, soil_centroid_y]
                
                angle_soil = int(calculate_angle_soil(left, top, right, bottom, nrow[2][0], nrow[2][1], 
                                nrow[3][0], nrow[3][1]))
                angle_rotate = int(calculate_angle_rotate(left, top, right, bottom, nrow[4][0], nrow[4][1], 
                                nrow[5][0], nrow[5][1]) % 180)
                if angle_rotate > 90:
                    angle_rotate = angle_rotate -180
                # angle_rotate = random_angle_and_difference(angle_rotate_temp)
                # row.append(f"{int(angle_rotate)}") 

                # csv_data.append(row) 
                
                if angle_rotate != 0 and (abs(angle_rotate) >= 10):
                    results_rows.append(angle_soil)
                    results_rows.append(angle_rotate)
                    ALL_results_rows.append(results_rows)
                    # Draw angle
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
                    cv2.line(img, (left, center_y), (right, center_y), (0, 0, 255), 2)
            
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
                # caption = f"{id}"
                # w, h = cv2.getTextSize(caption, 0, 1, 2)[0]     
                # cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
                # cv2.putText(img, caption, (left, top - 5), 0, 1, (255, 255, 255), 2, 16)
                # cv2.rectangle(img, (centerx - 3, centery - 33), (centerx + w + 10, centery), color, -1)
                # cv2.putText(img, caption, (centerx, centery - 5), 0, 1, (255, 255, 255), 2, 16)
            
            # 创建距离矩阵
            distance_matrix = create_distance_matrix(ALL_results_rows)
            
            # 使用蚁群算法规划路线
            best_route_indices, best_distance = ant_colony_optimization(distance_matrix)
            best_route = [ALL_results_rows[i] for i in best_route_indices]
            
            # 打印总距离
            print(f"总距离: {best_distance}")
            
            # 绘制路径
            for i in range(len(best_route)):
                loc = best_route[i]
                
                if i > 0:
                    prev_loc = best_route[i-1]
                    # cv2.line(img, tuple(prev_loc[1]), tuple(loc[1]), (255, 0, 255), 2)
                
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
                
            
            I += 1
            # Write data to CSV file
            with open("keypoints_data_" + str(I) + ".csv", mode="w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(csv_data)
                
            
            # Display the annotated frame
            # cv2.imshow(windows_name, img)
            cv2.imwrite(img_name, img)
            print("save done") 
            print("CSV file 'keypoints_data.csv' has been saved.")
            # print(os.path.join(root, file))
            
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


if __name__ == "__main__":

    # Replace 'your_directory' with the path to the directory you want to scan
    directory_path = r'D:\ultralytics-main\pose\valid\images'
    # 開始測量
    start = time.time()
    print_all_file_paths(directory_path)
    # 結束測量
    end = time.time()

    # 輸出結果
    print("執行時間：%f 秒" % (end - start))
    
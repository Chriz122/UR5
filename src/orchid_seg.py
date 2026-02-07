import cv2
import numpy as np
from hex2rgb import hex2rgb
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def calculate_area_and_centroid(points):
    """
    使用 OpenCV 計算多邊形的面積和幾何中心（質心）
    :param points: 多邊形頂點的列表，每個頂點由 (x, y) 坐標表示
    :return: (面積, 幾何中心坐標 (cx, cy))
    """
    # 將頂點數組轉換為 OpenCV 可接受的格式
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
    # 計算多邊形的面積
    area = cv2.contourArea(contour)
    
    # 計算輪廓的矩
    M = cv2.moments(contour)
    
    # 計算質心坐標
    if M["m00"] != 0:  # 確保面積不為零，避免除零錯誤
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # 如果面積為零，則無法計算質心
        raise ValueError("多邊形的面積為零，無法計算質心")
    
    return [cx, cy], area
    

def orchid_seg_predict(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    names_simple_chinese = ['人', '自行车', '汽车', '摩托', '飞机', '公交车', '火车', '卡车', '船', '交通灯',
                            '消防栓', '停车标志', '停车收费表', '长凳', '鸟', '猫', '狗', '马', '绵羊', '牛',
                            '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
                            '双滑雪板', '单滑雪板', '皮球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', 
                            '网球拍', '瓶子', '酒瓶', '杯子', '叉子', '刀子', '汤匙', '碗', '香蕉', '苹果', 
                            '三文治', '柳橙', '西兰花', '胡萝卜', '火腿肠', '披萨', '甜甜圈', '蛋糕', '椅子', '长椅',
                            '盆栽', '床', '餐桌', '厕所', '电视', '笔电', '鼠标', '遥控器', '键盘', '手机',
                            '微波驴', '烤箱', '烤土司机', '水槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊',
                            '吹风机', '牙刷'] # 要標示的目標物
    
    Yatsen = ['五片', '四片', '土下', '土上', '三片', '二片']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
    
    frame_PIL = Image.fromarray(img)
    draw = ImageDraw.Draw(frame_PIL, "RGBA")
    text_font = ImageFont.truetype('Arial_Unicode_MS.ttf', 30)
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []

    if (results[0].masks) is None:
        return None, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            color = hex2rgb(HEX[class_id % 20])
            xy = [tuple(point) for point in seg]

            if len(xy) >= 2:
                draw.polygon(xy, fill=(color[0], color[1], color[2], 128))

            # draw.rectangle((x, y, x2, y2), outline=color, width=5)
            # left, top, right, bottom = draw.textbbox((x + 5, y - 37), f'{objname} {score}', font=text_font)
            # draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=color)
            # draw.text((x + 5, y - 37), f'{objname} {score}', fill='white', font=text_font)
        
        return np.array(frame_PIL), img_name
    
    
def orchid_seg_predict_block(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    names_simple_chinese = ['人', '自行车', '汽车', '摩托', '飞机', '公交车', '火车', '卡车', '船', '交通灯',
                            '消防栓', '停车标志', '停车收费表', '长凳', '鸟', '猫', '狗', '马', '绵羊', '牛',
                            '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
                            '双滑雪板', '单滑雪板', '皮球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', 
                            '网球拍', '瓶子', '酒瓶', '杯子', '叉子', '刀子', '汤匙', '碗', '香蕉', '苹果', 
                            '三文治', '柳橙', '西兰花', '胡萝卜', '火腿肠', '披萨', '甜甜圈', '蛋糕', '椅子', '长椅',
                            '盆栽', '床', '餐桌', '厕所', '电视', '笔电', '鼠标', '遥控器', '键盘', '手机',
                            '微波驴', '烤箱', '烤土司机', '水槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊',
                            '吹风机', '牙刷'] # 要標示的目標物
    
    # Yatsen = ['五片', '四片', '土下', '土上', '三片', '二片']
    
    Yatsen = ['葉子', '土']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []

    if (results[0].masks) is None:
        return None, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            # color = hex2rgb(HEX[class_id % 20])
            xy = np.array([point for point in seg])
            
            if class_id == 0:
                color = [0, 255, 0]
            
            else:
                color = [0, 0, 0]

            if len(xy) >= 2:
                img = cv2.fillPoly(img, [xy], color=[color[0], color[1], color[2]])
        
        return img, img_name
    
    
def orchid_seg_leafs_number_predict_block(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    names_simple_chinese = ['人', '自行车', '汽车', '摩托', '飞机', '公交车', '火车', '卡车', '船', '交通灯',
                            '消防栓', '停车标志', '停车收费表', '长凳', '鸟', '猫', '狗', '马', '绵羊', '牛',
                            '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
                            '双滑雪板', '单滑雪板', '皮球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', 
                            '网球拍', '瓶子', '酒瓶', '杯子', '叉子', '刀子', '汤匙', '碗', '香蕉', '苹果', 
                            '三文治', '柳橙', '西兰花', '胡萝卜', '火腿肠', '披萨', '甜甜圈', '蛋糕', '椅子', '长椅',
                            '盆栽', '床', '餐桌', '厕所', '电视', '笔电', '鼠标', '遥控器', '键盘', '手机',
                            '微波驴', '烤箱', '烤土司机', '水槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊',
                            '吹风机', '牙刷'] # 要標示的目標物
    
    Yatsen = ['五片', '四片', '土', '三片', '二片']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.5, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []

    if (results[0].masks) is None:
        return img, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            color = hex2rgb(HEX[class_id % 20])
            xy = np.array([point for point in seg])
            
            if class_id == 2:
                color = [0, 0, 0]

            if len(xy) >= 2:
                img = cv2.fillPoly(img, [xy], color=[color[0], color[1], color[2]])
        
        return img, img_name
    
    
def orchid_seg_leafs_number_predict_block2(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    names_simple_chinese = ['人', '自行车', '汽车', '摩托', '飞机', '公交车', '火车', '卡车', '船', '交通灯',
                            '消防栓', '停车标志', '停车收费表', '长凳', '鸟', '猫', '狗', '马', '绵羊', '牛',
                            '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
                            '双滑雪板', '单滑雪板', '皮球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', 
                            '网球拍', '瓶子', '酒瓶', '杯子', '叉子', '刀子', '汤匙', '碗', '香蕉', '苹果', 
                            '三文治', '柳橙', '西兰花', '胡萝卜', '火腿肠', '披萨', '甜甜圈', '蛋糕', '椅子', '长椅',
                            '盆栽', '床', '餐桌', '厕所', '电视', '笔电', '鼠标', '遥控器', '键盘', '手机',
                            '微波驴', '烤箱', '烤土司机', '水槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊',
                            '吹风机', '牙刷'] # 要標示的目標物
    
    Yatsen = ['五片', '四片', '土', '三片', '二片']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.5, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []; results_row = [];

    if (results[0].masks) is None:
        return img, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            # color = hex2rgb(HEX[class_id % 20])
            color = [0, 255, 0]
            xy = np.array([point for point in seg])
            
            if len(xy) >= 2 and class_id == 0:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 5])
                
            elif len(xy) >= 2 and class_id == 1:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 4])
            
            elif class_id == 2:
                color = [0, 0, 0]
            
            elif len(xy) >= 2 and class_id == 3:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 3])
                
            elif len(xy) >= 2 and class_id == 4:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 2])

            if len(xy) >= 2:
                img = cv2.fillPoly(img, [xy], color=[color[0], color[1], color[2]])
        
        return img, img_name, results_row
                
                
                

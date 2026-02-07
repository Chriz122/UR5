import cv2
import torch
import numpy as np
import time
import pyrealsense2 as rs
import RW
from hex2rgb import hex2rgb
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

### 設定變數
refresh=True
H = 480
W = 640
C44=np.array([[16.6135509362135,-0.344070344779998,-0.526110782801237,-34.9417331227584],\
              [0.266554615611782,16.5800639041252,-0.378200734463540,-120.182235958949],\
              [-0.0304202992232481,-0.683616766513319,1.04416393594789,-215.47304299764],\
              [0,0,0,1]])
STOP = False
### ###

### intel 前處理(相關設定)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = ''
align_to = ''
align = ''

### ###

### yolo 導入 前處理
LABELS = open("C:/Users/Nuvo-7006DE-PoE/Desktop/tomato/tomato.names").read().strip().split("\n")
np.random.seed(666)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# 導入 YOLO 配置和權重文件并加載網路：
net = cv2.dnn_DetectionModel("C:/Users/Nuvo-7006DE-PoE/Desktop/tomato/yolov4-tiny-custom-tomato.cfg", "C:/Users/Nuvo-7006DE-PoE/Desktop/tomato/yolov4-tiny-custom-tomato_best.weights")
# 獲取 YOLO 未連接的輸出圖層
layer = net.getUnconnectedOutLayersNames()
### ###

def intel():
    global refresh,STOP,profile,align_to,align
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    try:
        while True:
            while(not refresh):
                time.sleep(0.1)
            frames_ = pipeline.wait_for_frames()
            aligned_frames = align.process(frames_)
            color_frame_ = aligned_frames.get_color_frame()
            if not color_frame_:
                continue
            color_image_ = np.asanyarray(color_frame_.get_data())
            cv2.imshow('real-time', color_image_)
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q') or STOP :
                break
    except:
        return True
    finally:
        pipeline.stop()
        cv2.destroyWindow('real-time')
        
def intelt():
    global refresh,STOP,profile,align_to,align
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    try:
        while True:
            while(not refresh):
                time.sleep(0.1)
            frames_ = pipeline.wait_for_frames()
            aligned_frames = align.process(frames_)
            color_frame_ = aligned_frames.get_color_frame()
            if not color_frame_:
                continue
            color_image_ = np.asanyarray(color_frame_.get_data())
            cv2.imshow('real-time', color_image_)

    except:
        return True     
      
def getXYZ():
    global refresh,C44,H,W,profile,align_to,align
    
    refresh=False
    print('getting data')
    temp=0
    depth_image = np.zeros((7,480,640))
    
    while temp!=7:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        # 對深度图黑洞區域進行填補
        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(aligned_depth_frame)
        depth_frame_modify = np.asanyarray(filled_depth.get_data())
        depth_image[temp,:,:] = depth_frame_modify
        temp+=1
    color_image = np.asanyarray(color_frame.get_data())
    refresh=True
    
    #####
    image = color_image
    RW.save_pic(image,"Bf_")
    depth_image=np.sort(depth_image,0)
    D=depth_image[3,:,:]
    
    # 從輸入圖像建構一個 blob，然後執行 YOLO 對象檢測器的前向傳遞，给我們BoundingBox和相關概率
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),\
                             swapRB=True, crop=False)
    net.setInput(blob)
    # 前向傳遞，獲得訊息
    layerOutputs = net.forward(layer)
    
    boxes = []; confidences = []; classIDs = [];
    xyz = []; kind = []; X = []; Y = []; Z = [];
    #####
    
    for output in layerOutputs:
        # 循环提取每个框
        for detection in output:
            # 提取當前目標類別 ID 和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # 通過確保檢測機率大於最小機率來過濾預測項目
            if confidence > 0.5:
                # 將邊界框座標相對於圖像大小進縮放，YOLO 返回的是BoundingBox的中心，
                # 后面是边界框的宽度和高度x,y座標
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 轉換出BoundingBox左上角座標
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # 更新BoundingBox座標、置信度和類別ID列表
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # 非最大值抑制，確定唯一BoundingBox
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 確定每一個boundingbox都有一個類別在
    if len(idxs) > 0:
        # 循環畫出邊框
        for i in idxs.flatten():
            # 提取座標和寬度
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX= int(x + w/2); centerY= int(y + h/2);
            xyz.append([ centerX, centerY, D[centerY-1,centerX-1], 1 ])
            kind.append(classIDs[i])
            # 畫出邊框和標籤
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,\
                0.5, color, 1, lineType=cv2.LINE_AA)
            #print('預測目標ID:'+ text +'  '+'目標中心座標:('+ str(centerX) +','+ str(centerY) +')')
    RW.save_pic(image,"Af_")
    cv2.imshow("Tag", image)
    cv2.waitKey(10)
    #####
    if len(idxs) > 0:
        xyz = np.array(xyz,'float32').T
        xyz[0,:] = xyz[0,:]*xyz[2,:]/10000
        xyz[1,:] = xyz[1,:]*xyz[2,:]/10000
        
        B=C44.dot(xyz)
        B[0,:] = B[0,:] - 45
        #B[1,:] = B[1,:] + 5
        B[2,:] = B[2,:] + 10
        B=B/1000
        X,Y,Z,_ = B.tolist()
    
    time.sleep(1)
    print('complete')
    return X,Y,Z,kind

def getXYZT(color_image,depth_image):
    global refresh,C44,H,W,profile,align_to,align
    
    # refresh=False
    print('getting data')
    # temp=0
    #depth_image = np.zeros((7,480,640))
    
    # while temp!=7:
    #     frames = pipeline.wait_for_frames()
    #     aligned_frames = align.process(frames)
    #     aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    #     color_frame = aligned_frames.get_color_frame()
    #     if not aligned_depth_frame or not color_frame:
    #         continue
    #     # 對深度图黑洞區域進行填補
    #     hole_filling = rs.hole_filling_filter()
    #     filled_depth = hole_filling.process(aligned_depth_frame)
    #     depth_frame_modify = np.asanyarray(filled_depth.get_data())
    #     depth_image[temp,:,:] = depth_frame_modify
    #     temp+=1
    # color_image = np.asanyarray(color_frame.get_data())
    # refresh=True
    
    #####
    image = color_image
    RW.save_pic(image,"Bf_")
    depth_image=np.sort(depth_image,0)
    D=depth_image
    
    # 從輸入圖像建構一個 blob，然後執行 YOLO 對象檢測器的前向傳遞，给我們BoundingBox和相關概率
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),\
                             swapRB=True, crop=False)
    net.setInput(blob)
    # 前向傳遞，獲得訊息
    layerOutputs = net.forward(layer)
    
    boxes = []; confidences = []; classIDs = [];
    xyz = []; kind = []; X = []; Y = []; Z = [];
    #####
    
    for output in layerOutputs:
        # 循环提取每个框
        for detection in output:
            # 提取當前目標類別 ID 和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # 通過確保檢測機率大於最小機率來過濾預測項目
            if confidence > 0.5:
                # 將邊界框座標相對於圖像大小進縮放，YOLO 返回的是BoundingBox的中心，
                # 后面是边界框的宽度和高度x,y座標
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 轉換出BoundingBox左上角座標
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # 更新BoundingBox座標、置信度和類別ID列表
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # 非最大值抑制，確定唯一BoundingBox
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 確定每一個boundingbox都有一個類別在
    if len(idxs) > 0:
        # 循環畫出邊框
        for i in idxs.flatten():
            # 提取座標和寬度
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centerX= int(x + w/2); centerY= int(y + h/2);
            xyz.append([ centerX, centerY, D[centerY-1,centerX-1], 1 ])
            kind.append(classIDs[i])
            # 畫出邊框和標籤
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,\
                0.5, color, 1, lineType=cv2.LINE_AA)
            print('預測目標ID:'+ text +'  '+'目標中心座標:('+ str(centerX) +','+ str(centerY) +')'+' '+'目標深度:'+str(D[centerY-1,centerX-1]))
    RW.save_pic(image,"Af_")
    cv2.imshow("Tag", image)
    cv2.waitKey(10)
    #####
    if len(idxs) > 0:
        xyz = np.array(xyz,'float32').T
        ppp = np.array(xyz)
        xyz[0,:] = xyz[0,:]*xyz[2,:]/10000
        xyz[1,:] = xyz[1,:]*xyz[2,:]/10000
        
        B=C44.dot(xyz)
        # B[0,:] = B[0,:] - 45
        # #B[1,:] = B[1,:] + 5
        # B[2,:] = B[2,:] + 10
        B[0,:] = B[0,:] 
        #B[1,:] = B[1,:] + 5
        B[2,:] = B[2,:] 
        B=B/1000
        X,Y,Z,_ = B.tolist()
    
    time.sleep(1)
    print('complete')
    return X,Y,Z,ppp,kind

def getXYZT_v8(color_image,depth_image):
    global refresh,C44,H,W,profile,align_to,align
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'] #色碼
    
    # refresh=False
    print('getting data')
    # temp=0
    #depth_image = np.zeros((7,480,640))
    
    # while temp!=7:
    #     frames = pipeline.wait_for_frames()
    #     aligned_frames = align.process(frames)
    #     aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    #     color_frame = aligned_frames.get_color_frame()
    #     if not aligned_depth_frame or not color_frame:
    #         continue
    #     # 對深度图黑洞區域進行填補
    #     hole_filling = rs.hole_filling_filter()
    #     filled_depth = hole_filling.process(aligned_depth_frame)
    #     depth_frame_modify = np.asanyarray(filled_depth.get_data())
    #     depth_image[temp,:,:] = depth_frame_modify
    #     temp+=1
    # color_image = np.asanyarray(color_frame.get_data())
    # refresh=True
    
    model = YOLO("weights/yolov8m-seg.pt")
    
    #####
    image = color_image
    RW.save_pic(image,"Bf_")
    depth_image=np.sort(depth_image,0)
    D=depth_image
    
    names_simple_chinese = ['人', '自行车', '汽车', '摩托', '飞机', '公交车', '火车', '卡车', '船', '交通灯',
                            '消防栓', '停车标志', '停车收费表', '长凳', '鸟', '猫', '狗', '马', '绵羊', '牛',
                            '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘',
                            '双滑雪板', '单滑雪板', '皮球', '风筝', '棒球棒', '棒球手套', '滑板', '冲浪板', 
                            '网球拍', '瓶子', '酒瓶', '杯子', '叉子', '刀子', '汤匙', '碗', '香蕉', '苹果', 
                            '三文治', '柳橙', '西兰花', '胡萝卜', '火腿肠', '披萨', '甜甜圈', '蛋糕', '椅子', '长椅',
                            '盆栽', '床', '餐桌', '厕所', '电视', '笔电', '鼠标', '遥控器', '键盘', '手机',
                            '微波驴', '烤箱', '烤土司机', '水槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊',
                            '吹风机', '牙刷'] # 要標示的目標物
    
    names_language = names_simple_chinese
    
    frame_PIL = Image.fromarray(image)
    
    draw = ImageDraw.Draw(frame_PIL, "RGBA")
    text_font = ImageFont.truetype('Arial_Unicode_MS.ttf',30)

    results = model.predict(source = image, verbose = False, device = 0, conf = 0.25, iou = 0.45)

    height, width, channels = image.shape
    
    # # 從輸入圖像建構一個 blob，然後執行 YOLO 對象檢測器的前向傳遞，给我們BoundingBox和相關概率
    # blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),\
    #                          swapRB=True, crop=False)
    # net.setInput(blob)
    # # 前向傳遞，獲得訊息
    # layerOutputs = net.forward(layer)
    
    boxes = []; confidences = []; classIDs = [];
    xyz = []; kind = []; X = []; Y = []; Z = [];
    #####
    segmentations = []

    if (results[0].masks) is None:
        pass
    
    else:

        for seg in results[0].masks.xyn:
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)
        
        # Get bboxs
        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)      
    
        # Visualize the results on the frame
        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]
        
            if objname in names_language:
                kind.append(class_id)
                color = hex2rgb(HEX[class_id%20])
                
                xy = []
                
                for i in range(0, len(seg)-1):
                    xy.append(tuple(seg[i]))
                    
                centerX = int((x + x2) / 2); 
                centerY = int((y + y2) / 2);
                
                xyz.append([ centerX, centerY, D[centerY-1,centerX-1], 1 ])
                kind.append(classIDs[i])

                if len(xy) >= 2 :
                    draw.polygon(xy, fill = (color[0], color[1], color[2], 128)) # 畫 segmentations 
                
                draw.rectangle((x, y, x2, y2), outline = color, width = 5)  # 畫 bbox 
                left, top, right, bottom = draw.textbbox((x+5, y-37), f'{objname} {score}', font = text_font)  # 畫 label            
                draw.rectangle((left-5, top-5, right+5, bottom+5), fill = color)  # 畫 label
                draw.text((x+5, y-37), f'{objname} {score}', fill = 'white', font = text_font )  # 畫 label
                
    
    color_frame_result = np.array(frame_PIL)
    
    RW.save_pic(image,"Af_")
    cv2.imshow("Tag", color_frame_result)
    cv2.waitKey(10)
    
    xyz = np.array(xyz,'float32').T
    ppp = np.array(xyz)
    xyz[0,:] = xyz[0,:]*xyz[2,:]/10000
    xyz[1,:] = xyz[1,:]*xyz[2,:]/10000
        
    B=C44.dot(xyz)
    # B[0,:] = B[0,:] - 45
    # #B[1,:] = B[1,:] + 5
    # B[2,:] = B[2,:] + 10
    B[0,:] = B[0,:] 
    #B[1,:] = B[1,:] + 5
    B[2,:] = B[2,:] 
    B=B/1000
    X,Y,Z,_ = B.tolist()
    
    time.sleep(1)
    print('complete')
    return X,Y,Z,ppp,kind
            
    
    # for output in layerOutputs:
    #     # 循环提取每个框
    #     for detection in output:
    #         # 提取當前目標類別 ID 和置信度
    #         scores = detection[5:]
    #         classID = np.argmax(scores)
    #         confidence = scores[classID]
    #         # 通過確保檢測機率大於最小機率來過濾預測項目
    #         if confidence > 0.5:
    #             # 將邊界框座標相對於圖像大小進縮放，YOLO 返回的是BoundingBox的中心，
    #             # 后面是边界框的宽度和高度x,y座標
    #             box = detection[0:4] * np.array([W, H, W, H])
    #             (centerX, centerY, width, height) = box.astype("int")
    #             # 轉換出BoundingBox左上角座標
    #             x = int(centerX - (width / 2))
    #             y = int(centerY - (height / 2))
                
    #             # 更新BoundingBox座標、置信度和類別ID列表
                
    #             boxes.append([x, y, int(width), int(height)])
    #             confidences.append(float(confidence))
    #             classIDs.append(classID)
                
    # # 非最大值抑制，確定唯一BoundingBox
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # # 確定每一個boundingbox都有一個類別在
    # if len(idxs) > 0:
    #     # 循環畫出邊框
    #     for i in idxs.flatten():
    #         # 提取座標和寬度
    #         (x, y) = (boxes[i][0], boxes[i][1])
    #         (w, h) = (boxes[i][2], boxes[i][3])
    #         centerX= int(x + w/2); centerY= int(y + h/2);
    #         xyz.append([ centerX, centerY, D[centerY-1,centerX-1], 1 ])
    #         kind.append(classIDs[i])
    #         # 畫出邊框和標籤
    #         color = [int(c) for c in COLORS[classIDs[i]]]
    #         cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
    #         text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    #         cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,\
    #             0.5, color, 1, lineType=cv2.LINE_AA)
    #         print('預測目標ID:'+ text +'  '+'目標中心座標:('+ str(centerX) +','+ str(centerY) +')'+' '+'目標深度:'+str(D[centerY-1,centerX-1]))
            
    # RW.save_pic(image,"Af_")
    # cv2.imshow("Tag", image)
    # cv2.waitKey(10)
    
    # #####
    # if len(idxs) > 0:
    #     xyz = np.array(xyz,'float32').T
    #     ppp = np.array(xyz)
    #     xyz[0,:] = xyz[0,:]*xyz[2,:]/10000
    #     xyz[1,:] = xyz[1,:]*xyz[2,:]/10000
        
    #     B=C44.dot(xyz)
    #     # B[0,:] = B[0,:] - 45
    #     # #B[1,:] = B[1,:] + 5
    #     # B[2,:] = B[2,:] + 10
    #     B[0,:] = B[0,:] 
    #     #B[1,:] = B[1,:] + 5
    #     B[2,:] = B[2,:] 
    #     B=B/1000
    #     X,Y,Z,_ = B.tolist()
    
    # time.sleep(1)
    # print('complete')
    # return X,Y,Z,ppp,kind


def intel_end():
    global STOP
    STOP = True
    RW.f_close()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import pyrealsense2 as rs
import RW

### 設定變數
refresh=True
H = 480
W = 640
C44=np.array([[16.6620633415338,0.298376613722848,-0.531117011262040,21.3476678413991],\
              [-0.287531697897276,16.7053096948173,-0.419938931526437,-82.4080705303489],\
              [0.0915137374049609,-0.0995437023903428,1.00445065391047,-299.671628036081],\
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
# net = cv2.dnn_DetectionModel("C:/Users/Nuvo-7006DE-PoE/Desktop/tomato/yolov4-tiny-custom-tomato.cfg", "C:/Users/Nuvo-7006DE-PoE/Desktop/tomato/yolov4-tiny-custom-tomato_best.weights")
# 獲取 YOLO 未連接的輸出圖層
# layer = net.getUnconnectedOutLayersNames()
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


def intel_end():
    global STOP
    STOP = True
    RW.f_close()
    cv2.destroyAllWindows()

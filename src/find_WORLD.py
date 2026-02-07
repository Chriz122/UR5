import cv2
import numpy as np

def find_WORLD(w, h, Chessboard, Chessboard_depth):

    # 找棋盘格角点
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001) # 阈值
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp*25  # 25 mm (棋盘格中每个格子的宽度)
    
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = [] # 在世界坐标系中的三维点
    imgpoints = [] # 在图像平面的二维点
    
    img = cv2.imread(Chessboard)
    c = [] 
    
    f = open(Chessboard_depth)
    
    lines = f.readlines()
    for line in lines:
        data = line.strip()
        if len(data) != 0:
           c.append(data.split())
    
    
    T = np.array([[16.6135509362135,-0.344070344779998,-0.526110782801237,-34.9417331227584],\
              [0.266554615611782,16.5800639041252,-0.378200734463540,-120.182235958949],\
              [-0.0304202992232481,-0.683616766513319,1.04416393594789,-215.47304299764],\
              [0,0,0,1]])
    
    W = [] # 世界座標
    
    # 获取画面中心点
    #获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        # 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #追加进入世界三维点和平面二维点中
        objpoints.append(objp)
        imgpoints.append(corners)
        
        for i in range(0, len(imgpoints[0])):
            
            u_index = int(imgpoints[0][i][0][1])
            u = float(imgpoints[0][i][0][1])
            
            v_index = int(imgpoints[0][i][0][0])
            v = float(imgpoints[0][i][0][0])
            
            Z = float(c[u_index][v_index])
            P = np.array([[v*Z/10000],
                          [u*Z/10000],
                          [Z],
                          [1]])
            
            W.append(np.dot(T,P)/1000)
            
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('find Corners'+str(Chessboard),cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners'+str(Chessboard),640,480)
        cv2.imshow('findCorners'+str(Chessboard),img)
        c=cv2.waitKey(2)
        
        return W
    
    
def find_WORLD_eyetohand(w, h, Chessboard, Chessboard_depth):

    # 找棋盘格角点
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001) # 阈值
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp*25  # 25 mm (棋盘格中每个格子的宽度)
    
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = [] # 在世界坐标系中的三维点
    imgpoints = [] # 在图像平面的二维点
    
    img = cv2.imread(Chessboard)
    c = [] 
    
    f = open(Chessboard_depth)
    
    lines = f.readlines()
    for line in lines:
        data = line.strip()
        if len(data) != 0:
           c.append(data.split())
    
    
    T = np.array([[-16.6810583932865, 0.00122180205434644, 0.539719877765054, 211.629140701675],
                    [-0.185780181830968, 16.5946293437513, -0.618152265854101, -573.454895826303],
                    [0.600135448393435, -0.229062082958058, 0.00504844060178511, 264.131650361031],
                    [0, 0, 0, 1]]) # D435i 轉移矩陣 eye to hand
    
    W = [] # 世界座標
    
    # 获取画面中心点
    #获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        # 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray,corners,(11, 11),(-1,-1),criteria)
        #追加进入世界三维点和平面二维点中
        objpoints.append(objp)
        imgpoints.append(corners)
        
        for i in range(0, len(imgpoints[0])):
            
            u_index = int(imgpoints[0][i][0][1])
            u = float(imgpoints[0][i][0][1])
            
            v_index = int(imgpoints[0][i][0][0])
            v = float(imgpoints[0][i][0][0])
            
            Z = float(c[u_index][v_index])
            P = np.array([[v*Z/10000],
                          [u*Z/10000],
                          [Z],
                          [1]])
            
            W.append(np.dot(T,P)/1000)
            
        print(W)
            
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('find Corners'+str(Chessboard),cv2.WINDOW_NORMAL)
        cv2.resizeWindow('find Corners'+str(Chessboard),640,480)
        cv2.imshow('find Corners'+str(Chessboard),img)
        c=cv2.waitKey(2)
        
        return W

if __name__ == '__main__':
    find_WORLD_eyetohand(4, 7, 'A1.bmp', 'D1.txt')
    
    # w = find_WORLD(6, 11, 'A1.bmp', 'D1.txt')
    # print(w)

    # for i in range(0, 66, 5):
    #     print(i)



    
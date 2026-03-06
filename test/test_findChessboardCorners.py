import cv2
import numpy as np

# 讀取圖片
img_path = '/home/jen-lab/Desktop/UR5/a20241107a/20260305_113053/A/A5.png'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not read image {img_path}")
else:
    # 轉換為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 棋盤格尺寸 (內角點數量)
    # 11x12 的棋盤格，內角點為 10x11
    board_size = (11, 12)
    
    # 尋找棋盤格角點
    # cv2.CALIB_CB_ADAPTIVE_THRESH: 使用自適應閾值
    # cv2.CALIB_CB_NORMALIZE_IMAGE: 正規化影像
    ret, corners = cv2.findChessboardCorners(gray, board_size, 
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        print(f"Found {len(corners)} corners.")
        
        # 次像素精確化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 繪製角點
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        
        # 顯示結果
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 輸出角點座標
        print("Corner coordinates (x, y):")
        for i, corner in enumerate(corners):
            print(f"Point {i+1}: ({corner[0][0]:.2f}, {corner[0][1]:.2f})")
    else:
        print(f"Could not find chessboard corners in {img_path}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        print(f"Expected board size: {board_size}")
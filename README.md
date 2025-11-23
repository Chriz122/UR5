# UR5 機器人視覺控制系統

## 專案描述

本專案是一個整合 UR5 機器人手臂與視覺系統的應用程式，主要用於物體檢測、姿態估計、機器人控制和路徑優化。系統結合了 YOLO 深度學習模型、Intel RealSense D435i 深度相機，以及螞蟻殖民優化算法，實現了自動化物體抓取和操作任務。

## 主要功能

### 1. 機器人控制 (`function_arm.py`)
- 連接 UR5 機器人手臂
- 實現機器人移動控制（線性移動、關節移動）
- 獲取機器人當前位置和姿態
- 控制夾爪開合

### 2. 視覺處理 (`function_intelFORyolo.py`, `orchid_pose_d435.py`)
- 使用 Intel RealSense D435i 相機獲取彩色和深度影像
- 實現影像對齊和深度數據處理
- 蘭花姿態估計
- 顏色識別和區域分割

### 3. 校準系統 (`UR5_calibration.py`, `find_WORLD.py`)
- 機器人與相機之間的眼在手上（eye-to-hand）校準
- 棋盤格角點檢測
- 世界座標系與相機座標系的轉換
- 數據收集和儲存

### 4. YOLO 物體檢測 (`YOLO_detect+rotate.py`)
- 整合 YOLO 模型進行實時物體檢測
- 物體位置計算和追蹤
- 機器人自動抓取控制

### 5. 路徑優化 (`algorithm/ant.py`)
- 螞蟻殖民優化算法實現
- 多點路徑規劃和優化
- 距離矩陣計算

## 理論基礎

### 1. 手臂標定 (Eye-to-Hand Calibration)

眼在手上校準用於建立相機座標系與機器人基座標系之間的關係。通過多組已知機器人姿態和對應的棋盤格影像，求解轉移矩陣：

```
P_robot = T_eye_to_hand * P_camera
```

其中：
- P_robot: 機器人基座標系中的點
- P_camera: 相機座標系中的點
- T_eye_to_hand: 11*12 轉移矩陣

校準過程使用 Tsai 的方法，通過最小二乘法優化轉移矩陣參數。

### 2. 深度影像處理 (未直接使用RealSense 內建3D 座標轉換)

Intel RealSense D435i 提供 RGB-D 數據，系統使用以下技術處理深度資訊：

- **影像對齊**: 將深度影像對齊到彩色影像，消除視差
- **孔洞填充**: 使用臨近像素補充無效深度值
- **座標轉換**: 將像素座標轉換為 3D 世界座標

深度到 3D 座標的轉換公式：

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_value * depth_scale
```

其中 (fx, fy) 為焦距，(cx, cy) 為光心座標。

### 3. 螞蟻殖民優化算法 (Ant Colony Optimization)

ACO 是一種受螞蟻覓食行為啟發的元啟發式算法，用於解決組合優化問題。算法原理：

- **費洛蒙更新**: 螞蟻在路徑上釋放費洛蒙，較短路徑費洛蒙濃度更高
- **概率選擇**: 螞蟻根據費洛蒙濃度和啟發式資訊選擇下一節點

```
P_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
```

其中：
- τ_ij: 邊 (i,j) 上的費洛蒙量
- η_ij: 邊 (i,j) 的啟發式值 (通常為 1/d_ij)
- α, β: 控制參數

系統使用 ACO 優化多點抓取序列，減少總移動距離。

## 系統架構

```
UR5/
├── requirement.txt          # Python 依賴包列表
├── data/                    # 數據儲存目錄
│   ├── images/             # 校準影像
│   └── txts/               # 深度數據和姿態數據
├── models/                 # 訓練好的模型文件
└── src/                    # 源代碼
    ├── calculate_transition_matrix.py  # 轉移矩陣計算（空文件）
    ├── find_WORLD.py       # 世界座標檢測
    ├── function_arm.py     # 機器人控制函數
    ├── function_intelFORyolo.py  # Intel 相機處理
    ├── orchid_pose_d435.py # 蘭花姿態估計
    ├── UR5_calibration.py  # 機器人校準
    ├── YOLO_detect+rotate.py  # 主控制腳本
    └── algorithm/
        └── ant.py          # 螞蟻優化算法
```

## 使用說明

### 1. 系統初始化

首先運行主控制腳本：
```python
python src/YOLO_detect+rotate.py
```

### 2. 校準過程

運行校準腳本進行眼在手上校準：
```python
python src/UR5_calibration.py
```

校準步驟：
1. 移動機器人到不同位置
2. 按 'x' 鍵拍攝棋盤格影像
3. 收集足夠的校準數據點
4. 計算轉移矩陣

### 3. 物體檢測和抓取

1. 啟動 YOLO 檢測：
   - 按 'Q' 開始檢測
   - 系統會自動檢測物體位置

2. 自動抓取：
   - 系統計算物體 3D 位置
   - 機器人移動到抓取位置
   - 執行抓取動作

### 4. 路徑優化

使用螞蟻算法進行多點路徑規劃：
```python
from algorithm.ant import ant_colony_optimization
route, distance, time = ant_colony_optimization(distance_matrix)
```

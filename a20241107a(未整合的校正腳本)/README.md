# 專案 README

此專案包含 MATLAB 腳本，用於相機校準、座標轉換以及與機器人視覺相關的繪圖功能。

## 檔案樹狀結構

```
a20241107a/
├── A/*.png                    # 校準影像 1 ~ 5
├── change_uvZtoXWYXZW.m      # 將影像點轉換為手臂座標 (MATLAB)
├── change_uvZtoXWYXZW.py     # 將影像點轉換為手臂座標 (Python)
├── D/*.txt                   # 深度資料 1 ~ 5
├── depth_image_*.png # 深度影像 1 ~ 5
├── eye_to_hand_calibration_new2.m # 眼到手校準腳本 (MATLAB)
├── eye_to_hand_calibration_new2.py # 眼到手校準腳本 (Python)
├── plotCalbPoint.m           # 繪製 2D 校準點 (MATLAB)
├── plotCalbPoint.py          # 繪製 2D 校準點 (Python)
├── plotCalbPoint3d.m         # 繪製 3D 校準點（帶標籤）(MATLAB)
├── plotCalbPoint3d.py        # 繪製 3D 校準點（帶標籤）(Python)
├── plotCalbPoint3d_noText.m  # 繪製 3D 校準點（無標籤）(MATLAB)
├── plotCalbPoint3d_noText.py # 繪製 3D 校準點（無標籤）(Python)
├── plotCalbPointErr.m        # 視覺化校準誤差 (MATLAB)
├── plotCalbPointErr.py       # 視覺化校準誤差 (Python)
├── pose.txt                  # 姿態資訊
├── test.m                    # 測試腳本（讀取姿態並插值）(MATLAB)
├── test.py                   # 測試腳本（讀取姿態並插值）(Python)
├── Untitled.m                # 設定校準點位置 (MATLAB)
└── Untitled.py               # 設定校準點位置 (Python)
```

## MATLAB 檔案 (.m) 詳細說明

### change_uvZtoXWYXZW.m
- **目的**：將影像點 (u,v) 與深度 Z 轉換為世界座標 (X,Y,Z,W)，使用轉換矩陣。
- **功能**：
  - 讀取影像 (A1.png) 和深度資料 (D1.txt)。
  - 在影像中檢測棋盤格點。
  - 應用座標轉換將像素座標轉換為世界座標。
  - 輸出最小和最大 Z 值。
- **依賴項**：需要影像檔案 'A1.png' 和深度檔案 'D1.txt'。

### eye_to_hand_calubration_new2.m
- **目的**：使用多個校準影像執行眼到手相機校準。
- **功能**：
  - 在 3D 空間中設定校準點。
  - 載入多個影像 (A1.png 到 A5.png) 並檢測棋盤格點。
  - 準備相機校準過程的資料。
- **依賴項**：需要影像檔案 'A1.png' 到 'A5.png'。

### plotCalbPoint.m
- **目的**：繪製帶有數值標籤的 2D 校準點。
- **功能**：接收 xy 座標並將其繪製為 '+' 標記與文字標籤。
- **參數**：xy - [x,y] 點陣列。

### plotCalbPoint3d.m
- **目的**：繪製帶有數值標籤和軸標籤的 3D 校準點。
- **功能**：接收 xyz 座標並建立帶有文字標籤的 3D 散點圖。
- **參數**：xy - [x,y,z] 點陣列。

### plotCalbPoint3d_noText.m
- **目的**：繪製不帶文字標籤的 3D 校準點。
- **功能**：類似於 plotCalbPoint3d 但點上無數值標籤。
- **參數**：xy - [x,y,z] 點陣列。

### plotCalbPointErr.m
- **目的**：以網格格式視覺化校準誤差，並使用顏色編碼。
- **功能**：
  - 在不同子圖中繪製 X、Y、Z 軸的誤差。
  - 根據誤差大小進行顏色編碼（黑色 ≤1mm，藍色 ≤10mm，紅色 ≤20mm，洋紅色 >20mm）。
- **參數**：err - 誤差陣列，rn - 行數，cn - 列數。

### test.m
- **目的**：測試腳本，用於讀取姿態資料並執行雙線性插值。
- **功能**：
  - 從 'pose - 複製.txt' 讀取姿態資料。
  - 在 11x12 網格上執行雙線性插值。
  - 準備插值的 X、Y、Z 座標。
- **依賴項**：需要 'pose - 複製.txt' 檔案。

### Untitled.m
- **目的**：為多個位置設定 3D 校準點位置。
- **功能**：為 4 個不同高度產生相機位置陣列，每個高度有 132 個點。
- **輸出**：cam_location 陣列，包含 [X,Y,Z] 座標。

## Python 檔案 (.py) 詳細說明

### change_uvZtoXWYXZW.py
- **目的**：將影像點 (u,v) 與深度 Z 轉換為世界座標 (X,Y,Z,W)，使用轉換矩陣。
- **功能**：
  - 讀取影像 (A1.png) 和深度資料 (D1.txt)。
  - 在影像中檢測棋盤格點。
  - 應用座標轉換將像素座標轉換為世界座標。
  - 輸出最小和最大 Z 值。
- **依賴項**：需要影像檔案 'A1.png' 和深度檔案 'D1.txt'。需要 OpenCV 和 NumPy。

### eye_to_hand_calubration_new2.py
- **目的**：使用多個校準影像執行眼到手相機校準。
- **功能**：
  - 設定校準點在 3D 空間中。
  - 載入多個影像 (A1.png 到 A5.png) 並檢測棋盤格點。
  - 準備相機校準過程的資料。
- **依賴項**：需要影像檔案 'A1.png' 到 'A5.png'，深度檔案 'D1.txt' 到 'D5.txt'。需要 OpenCV, NumPy, SciPy。

### plotCalbPoint.py
- **目的**：繪製帶有數值標籤的 2D 校準點。
- **功能**：接收 xy 座標並將其繪製為 '+' 標記與文字標籤。
- **參數**：xy - [x,y] 點陣列。

### plotCalbPoint3d.py
- **目的**：繪製帶有數值標籤和軸標籤的 3D 校準點。
- **功能**：接收 xyz 座標並建立帶有文字標籤的 3D 散點圖。
- **參數**：xy - [x,y,z] 點陣列。

### plotCalbPoint3d_noText.py
- **目的**：繪製不帶文字標籤的 3D 校準點。
- **功能**：類似於 plotCalbPoint3d 但點上無數值標籤。
- **參數**：xy - [x,y,z] 點陣列。

### plotCalbPointErr.py
- **目的**：以網格格式視覺化校準誤差，並使用顏色編碼。
- **功能**：
  - 在不同子圖中繪製 X、Y、Z 軸的誤差。
  - 根據誤差大小進行顏色編碼（黑色 ≤1mm，藍色 ≤10mm，紅色 ≤20mm，洋紅色 >20mm）。
- **參數**：err - 誤差陣列，rn - 行數，cn - 列數。

### test.py
- **目的**：測試腳本，用於讀取姿態資料並執行雙線性插值。
- **功能**：
  - 從 'pose - 複製.txt' 讀取姿態資料。
  - 在 11x12 網格上執行雙線性插值。
  - 準備插值的 X、Y、Z 座標。
- **依賴項**：需要 'pose - 複製.txt' 檔案。

### Untitled.py
- **目的**：設定 3D 校準點位置為多個位置。
- **功能**：為 4 個不同高度產生相機位置陣列，每個高度有 132 個點。
- **輸出**：cam_location 陣列，包含 [X,Y,Z] 座標。

## 資料檔案
- D1.txt 到 D5.txt：深度資料檔案
- pose.txt：姿態資訊
- A1.png 到 A5.png：校準影像
- depth_image_*.png：深度影像

## 使用方式
在 MATLAB 中執行校準腳本，需要 Computer Vision Toolbox 以使用棋盤格檢測功能。

在 Python 中執行，需要安裝 OpenCV, NumPy, Matplotlib, SciPy 等套件。
clc; clear; close all;

% 打開檔案並讀取數據
fileID = fopen('pose - 複製.txt', 'r');
if fileID == -1
    error('無法打開文件，請檢查路徑是否正確');
end

% 讀取檔案中的數據（假設每行有9列數據，前三列為XYZ，後六列為其他數據）
data = textscan(fileID, '%f %f %f %f %f %f %f %f %f', 'Delimiter', ' ');
fclose(fileID);

% 確保檔案有四行數據
if length(data{1}) < 4
    error('數據文件中的行數不足，無法使用四個角點');
end

% 提取四個角點座標 (第一列到第三列為XYZ)
X_corners = [data{1}(1), data{1}(2), data{1}(3), data{1}(4)];
Y_corners = [data{2}(1), data{2}(2), data{2}(3), data{2}(4)];
Z_corners = [data{3}(1), data{3}(2), data{3}(3), data{3}(4)];

% 提取每個角點對應的後六列數據
extra_data = [data{4}(1:4), data{5}(1:4), data{6}(1:4), data{7}(1:4), data{8}(1:4), data{9}(1:4)];

% 設定分佈的行數與列數 (X 方向為 11，Y 方向為 12)
rows = 12; % Y 方向上的點數
cols = 11; % X 方向上的點數

% 創建 X, Y, Z 的網格
[X_grid, Y_grid] = meshgrid(linspace(0, 1, cols), linspace(0, 1, rows));

% 計算網格點的 X, Y, Z 座標，根據角點進行插值
X = (1 - X_grid) .* (1 - Y_grid) * X_corners(1) + ...
    X_grid .* (1 - Y_grid) * X_corners(2) + ...
    X_grid .* Y_grid * X_corners(4) + ...
    (1 - X_grid) .* Y_grid * X_corners(3);

Y = (1 - X_grid) .* (1 - Y_grid) * Y_corners(1) + ...
    X_grid .* (1 - Y_grid) * Y_corners(2) + ...
    X_grid .* Y_grid * Y_corners(4) + ...
    (1 - X_grid) .* Y_grid * Y_corners(3);

Z = (1 - X_grid) .* (1 - Y_grid) * Z_corners(1) + ...
    X_grid .* (1 - Y_grid) * Z_corners(2) + ...
    X_grid .* Y_grid * Z_corners(4) + ...
    (1 - X_grid) .* Y_grid * Z_corners(3);

% 從原始四個角點的附加數據進行插值，以構建對應的後六列數據
extra_cols = zeros(rows * cols, 6);
for i = 1:6
    extra_cols(:, i) = (1 - X_grid(:)) .* (1 - Y_grid(:)) * extra_data(1, i) + ...
        X_grid(:) .* (1 - Y_grid(:)) * extra_data(2, i) + ...
        X_grid(:) .* Y_grid(:) * extra_data(4, i) + ...
        (1 - X_grid(:)) .* Y_grid(:) * extra_data(3, i);
end

% 將所有數據轉換為一個 132x9 的矩陣，並將其轉換為先行後列的順序
points = [X(:), Y(:), Z(:), extra_cols];

% 確保數據按照行優先的順序排列
points = reshape(points, rows, cols, 9);  % 將數據重塑為 (rows, cols, 9)
points = permute(points, [2, 1, 3]);      % 交換行和列，使得行優先
points = reshape(points, [], 9);          % 將數據展平為132x9的矩陣

% 將數據寫入名為 'pose.txt' 的文件中，並覆蓋舊文件
fileID = fopen('pose.txt', 'w');  % 以 'w' 模式打開文件將覆蓋舊文件
for i = 1:size(points, 1)
    fprintf(fileID, '%f %f %f %f %f %f %f %f %f\n', points(i, :));
end
fclose(fileID);

disp('文件 pose.txt 已成功生成並覆蓋舊文件，包含 132 行，每行 9 列數字。');

% 繪製出 3D 點和連接的平面
figure;
scatter3(points(:, 1), points(:, 2), points(:, 3), 'filled');  % 使用插值後的點繪製散點圖
hold on;
mesh(reshape(points(:, 1), cols, rows)', reshape(points(:, 2), cols, rows)', reshape(points(:, 3), cols, rows)');  % 繪製網格
xlabel('X');
ylabel('Y');
zlabel('Z');
title('11x12 平均分佈座標點的 3D 網格');
grid on;
hold off;

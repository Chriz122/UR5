import cv2
import numpy as np
import math
import pandas as pd
import os

# Define the paths
original_csv_path = 'keypoints_data_1.csv'
new_csv_path = 'keypoints_data_new.csv'
img_path = 'predict-pose-seg-single1.jpg'

# Load CSV data if the original file exists
if os.path.exists(original_csv_path):
    csv_data = pd.read_csv(original_csv_path)
    
    # Ensure the 'angle' column exists
    if 'angle' not in csv_data.columns:
        csv_data['angle'] = None

    # Filter out invalid points and save to new CSV
    def filter_invalid_points(csv_data):
        valid_data = []
        for index, row in csv_data.iterrows():
            try:
                x3_str, y3_str = row['(x3, y3)'], row['(x4, y4)']
                if isinstance(x3_str, str) and isinstance(y3_str, str):
                    x3, y3 = map(int, x3_str.strip('()').split(','))
                    x4, y4 = map(int, y3_str.strip('()').split(','))
                    if (x3 != 0 or y3 != 0) and (x4 != 0 or y4 != 0):
                        valid_data.append(row)
                    else:
                        csv_data.at[index, 'angle'] = '無效'
            except (ValueError, AttributeError):
                csv_data.at[index, 'angle'] = '無效'

        valid_data = pd.DataFrame(valid_data)
        valid_data.to_csv(new_csv_path, index=False)
        os.remove(original_csv_path)  # Delete the original CSV file
        return new_csv_path

    valid_csv_path = filter_invalid_points(csv_data)
else:
    valid_csv_path = new_csv_path

# Load the new CSV data
if not os.path.exists(valid_csv_path):
    print(f"File {valid_csv_path} does not exist. Exiting.")
    exit()

valid_csv_data = pd.read_csv(valid_csv_path)

# Extract points and IDs for the first line from the valid CSV
def extract_line1_points_from_csv(csv_data):
    line1_points = []
    ids = []
    for index, row in csv_data.iterrows():
        try:
            x3_str, y3_str = row['(x3, y3)'], row['(x4, y4)']
            if isinstance(x3_str, str) and isinstance(y3_str, str):
                x3, y3 = map(int, x3_str.strip('()').split(','))
                x4, y4 = map(int, y3_str.strip('()').split(','))
                if (x3 != 0 or y3 != 0) and (x4 != 0 or y4 != 0):
                    if pd.isna(row['angle']):
                        line1_points.append(((x3, y3), (x4, y4)))
                        ids.append(row['id'])
        except (ValueError, AttributeError):
            pass
    return line1_points, ids

# Global variables to store points and angles
points = []
line1_points, ids = extract_line1_points_from_csv(valid_csv_data)

# Find the next index to calculate based on the smallest uncalculated id
def find_next_index(csv_data, ids):
    uncalculated_ids = [id for id in ids if pd.isna(csv_data.loc[csv_data['id'] == id, 'angle']).values[0]]
    if uncalculated_ids:
        next_id = min(uncalculated_ids)
        return ids.index(next_id)
    return len(csv_data)

current_index = find_next_index(valid_csv_data, ids)

def calculate_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf')  # handle vertical lines
    return (y2 - y1) / (x2 - x1)

def are_parallel_within_tolerance(slope1, slope2, tolerance=10):
    if slope1 == float('inf') and slope2 == float('inf'):
        return True
    if slope1 == float('inf') or slope2 == float('inf'):
        return False
    angle_radians = math.atan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees < tolerance

def angle_between_lines(slope1, slope2):
    if slope1 == float('inf'):
        return 90 - math.degrees(math.atan(slope2))
    if slope2 == float('inf'):
        return 90 - math.degrees(math.atan(slope1))
    angle_radians = math.atan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    return math.degrees(angle_radians)

def mouse_callback(event, x, y, flags, param):
    global points, img, current_index, valid_csv_data
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2 and current_index < len(line1_points):
            current_line1 = line1_points[current_index]
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)
            slope1 = calculate_slope(*current_line1[0], *current_line1[1])
            slope2 = calculate_slope(*points[0], *points[1])
            if are_parallel_within_tolerance(slope1, slope2):
                result_text = f"ID {ids[current_index]}: 兩條直線是平行的。"
                angle = 0 - (angle_between_lines(slope1, slope2))  # Parallel lines have 0 degrees difference
            else:
                angle = angle_between_lines(slope1, slope2)
                result_text = f"ID {ids[current_index]}: 兩條直線不是平行的，它們之間的夾角是 {angle:.2f} 度。"
            valid_csv_data.at[valid_csv_data[valid_csv_data['id'] == ids[current_index]].index[0], 'angle'] = angle
            valid_csv_data.to_csv(valid_csv_path, index=False)
            print(result_text)
            # cv2.putText(img, result_text, (10, 30 + current_index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            points = []
            current_index = find_next_index(valid_csv_data, ids)

if __name__ == "__main__":
    # Load image
    img = cv2.imread(img_path)

    # Initialize OpenCV window
    cv2.namedWindow("Draw Lines")
    cv2.setMouseCallback("Draw Lines", mouse_callback)

    # Draw lines for IDs that haven't been calculated
    for line_points, id in zip(line1_points, ids):
        if pd.isna(valid_csv_data.at[valid_csv_data[valid_csv_data['id'] == id].index[0], 'angle']):
            # Draw line
            cv2.line(img, line_points[0], line_points[1], (255, 0, 0), 2)
            # Calculate midpoint
            midpoint = ((line_points[0][0] + line_points[1][0]) // 2, (line_points[0][1] + line_points[1][1]) // 2)
            # Draw ID at midpoint
            cv2.putText(img, str(id), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    while current_index < len(line1_points):
        cv2.imshow("Draw Lines", img)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()


































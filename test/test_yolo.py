from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2

# model = YOLO("/home/jen-lab/Desktop/UR5/models/best_sick_keypoint.v14i.yolo-master-UltraOptimizedMoE-shuffle-pose-s_v0_2_dinov3_hardsample_two_points.onnx")
model = YOLO("/home/jen-lab/Desktop/UR5/models/best.pt")

 # Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
lign_to = rs.stream.color
align = rs.align(align_to)
temp=1 #從多少開始
# Streaming loop

while True:
    key = cv2.waitKeyEx(10)
    if key == 27: #按下 Esc 關閉視窗
        cv2.destroyAllWindows()
        break

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
     # frames.get_depth_frame() is a 640x360 depth image

     # Align the depth frame to color frame
    aligned_frames = align.process(frames)

     # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

     # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue
     # 對深度图黑洞區域進行填補
    hole_filling = rs.hole_filling_filter()
    filled_depth = hole_filling.process(aligned_depth_frame)   
    depth_frame_modify = np.asanyarray(filled_depth.get_data())
    colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame_modify, alpha=0.03), cv2.COLORMAP_JET)

    depth_image = depth_frame_modify
    color_image = np.asanyarray(color_frame.get_data())    

    color_image=np.uint8(color_image)
    color_image_copy = color_image.copy()

    # depth_image=np.uint8(depth_image)
    # depth_image[depth_image!=0]=depth_image[depth_image!=0]%200+55
        

    # YOLO 預測
    results = model(color_image, verbose=False, conf=0.25, iou=0.45)

    dc_images = np.hstack((results[0].plot(), color_image))
    cv2.imshow("test", dc_images)
    cv2.waitKey(1)

# original = cv2.imread("original.jpg")
# results = model(original, verbose=False, conf=0.03, iou=0.45)

# dc_images = np.hstack((results[0].plot(), original))
# cv2.imshow("test", dc_images)
# cv2.waitKey(0)
                                                            
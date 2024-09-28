from ultralytics import YOLO

#model = YOLO('models/yolo5_last.pt') #for tennis balls
#model = YOLO('models/yolo5_last.pt') #for tennis balls

model = YOLO('models/yolov8x') # for tracking

result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)


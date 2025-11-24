from ultralytics import YOLO

model = YOLO("/Users/dhruvkulshrestha/Desktop/Tennis_analysis /keypoint_model/keypoints_model-2.pth.pt")  

results = model.predict(source="input_videos/input_video.mp4", save=True, conf=0.5, save_txt=True)
print(results)

bboxes = results[0].boxes.xyxy.tolist()  # Extract bounding box coordinates
print('bboxes:', bboxes)
for box in bboxes:
    x1, y1, x2, y2 = box
    print(f"Bounding Box - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

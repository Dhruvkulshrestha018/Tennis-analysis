import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = models.resnet50(weights=None)  
model.fc = torch.nn.Linear(model.fc.in_features, 28) 


model.load_state_dict(torch.load("keypoints_model.pth", map_location=device, weights_only=False))
model.to(device)
model.eval()

print("Model loaded successfully!")
print(f"Model output dimension: {model.fc.out_features}")
print(f"Expected: 14 keypoints × 2 coordinates = 28 values")


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_keypoints(frame):
    """
    Predict keypoints for a single frame
    """
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        keypoints = model(input_tensor)
    
    
    keypoints = keypoints.cpu().numpy()[0]
    
    
    h, w = frame.shape[:2]
    keypoints[::2] *= w / 224.0  # x coordinates
    keypoints[1::2] *= h / 224.0  # y coordinates
    
    return keypoints

def visualize_keypoints(frame, keypoints, confidence_threshold=0.1):
    """
    Draw keypoints on the frame with confidence-based coloring
    """
    h, w = frame.shape[:2]
    
    
    for i in range(0, len(keypoints), 2):
        x, y = int(keypoints[i]), int(keypoints[i+1])
        
        
        if 0 <= x < w and 0 <= y < h:
            # You could add confidence-based coloring here if your model outputs confidence
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)  
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)     
            cv2.putText(frame, str(i//2), (x+8, y+8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def test_with_image(image_path):
    """
    Test the model with a single image
    """
    print(f"Testing with image: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        print("Please make sure the image path is correct.")
        return
    
    print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
    
    
    keypoints = predict_keypoints(frame)
    
    print(f" Predicted {len(keypoints)//2} keypoints")
    print(" Keypoints coordinates:")
    for i in range(0, len(keypoints), 2):
        print(f"  Keypoint {i//2}: ({keypoints[i]:.1f}, {keypoints[i+1]:.1f})")
    
    
    frame_with_keypoints = visualize_keypoints(frame.copy(), keypoints)
    
    
    cv2.imshow('Tennis Court Keypoints - Press any key to close', frame_with_keypoints)
    print("Displaying image with keypoints...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    output_path = "output_with_keypoints.jpg"
    cv2.imwrite(output_path, frame_with_keypoints)
    print(f"💾 Result saved as: {output_path}")

def test_with_video(video_path, output_path="output_video.mp4"):
    """
    Test the model with a video file
    """
    print(f"Testing with video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Could not open video file: {video_path}")
        print("Please make sure the video path is correct.")
        return
    
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎥 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("⏳ Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        
        keypoints = predict_keypoints(frame)
        
       
        frame_with_keypoints = visualize_keypoints(frame, keypoints)
        
        
        cv2.putText(frame_with_keypoints, f"Frame: {frame_count}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        
        out.write(frame_with_keypoints)
        
        
        frame_count += 1
        if frame_count % 30 == 0:  
            print(f"📹 Processed {frame_count}/{total_frames} frames")
        
        
        cv2.imshow('Tennis Court Keypoints - Press Q to quit', frame_with_keypoints)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Processing stopped by user")
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete!")
    print(f" Output saved as: {output_path}")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    print(" Tennis Court Keypoint Detection")
    print("=" * 40)
    
    
    test_image_path = "/Users/dhruvkulshrestha/Desktop/Tennis_analysis /input_videos/image.png" 
    
    if os.path.exists(test_image_path):
        test_with_image(test_image_path)
    else:
        print(f"⚠️  Test image not found at: {test_image_path}")
        print("Please update the test_image_path variable with your image path.")
    
    
    print("\n" + "=" * 40)
    video_path = "/Users/dhruvkulshrestha/Desktop/Tennis_analysis /input_videos/input_video.mp4"  
    
    if os.path.exists(video_path):
        choice = input("Do you want to process the video? (y/n): ").lower().strip()
        if choice == 'y':
            test_with_video(video_path)
    else:
        print(f"  Video file not found at: {video_path}")
        print("Please update the video_path variable with your video path.")
    
    print(" Program completed!")
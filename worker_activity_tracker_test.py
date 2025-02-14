import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Initialize YOLOv8 model (human detection)
model = YOLO('yolov8n.pt')  # Nano version for speed

# Activity tracking parameters
MOVEMENT_THRESHOLD = 15  # Pixels movement between frames
ACTIVITY_WINDOW = 30    # Analyze last frames

# Store worker positions and activity
worker_tracks = defaultdict(lambda: {
    'positions': [],
    'activity': False
})

def calculate_movement(positions):
    """Calculate total movement in recent frames"""
    if len(positions) < 2:
        return 0
    return np.sqrt((positions[-1][0]-positions[-2][0])**2 + 
                   (positions[-1][1]-positions[-2][1])**2)

def process_frame(frame):
    """Process single video frame"""
    # Detect humans
    results = model.track(frame, persist=True, classes=0)  # Class 0 = person
    boxes = results[0].boxes.xywh.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

    # Update tracks and check activity
    for box, track_id in zip(boxes, ids):
        x, y, w, h = box
        center = (int(x), int(y))
        
        # Store last positions
        worker_tracks[track_id]['positions'].append(center)
        if len(worker_tracks[track_id]['positions']) > ACTIVITY_WINDOW:
            worker_tracks[track_id]['positions'].pop(0)
        
        # Calculate movement
        movement = calculate_movement(worker_tracks[track_id]['positions'])
        worker_tracks[track_id]['activity'] = movement > MOVEMENT_THRESHOLD
        
        # Draw visualization
        color = (0, 255, 0) if worker_tracks[track_id]['activity'] else (0, 0, 255)
        cv2.rectangle(frame, 
                      (int(x - w/2), int(y - h/2)),
                      (int(x + w/2), int(y + h/2)), 
                      color, 2)
        cv2.putText(frame, f"ID: {int(track_id)}", 
                    (int(x - w/2), int(y - h/2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Main processing loop
cap = cv2.VideoCapture('factory_worker.mp4')  # Input video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    
    # Show results
    cv2.imshow('Factory Monitor', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
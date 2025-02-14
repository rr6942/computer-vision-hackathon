import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Initialize YOLOv8 model (human detection)
model = YOLO('yolov8n.pt')  # Nano version for speed

# Activity tracking parameters
MOVEMENT_THRESHOLD = 1  # Pixels movement between frames
ACTIVITY_WINDOW = 30    # Analyze last frames

# Store worker positions and activity
worker_tracks = defaultdict(lambda: {
    'positions': [],
    'activity': False,
    'active_frames': 0,
    'total_frames': 0
})

def calculate_movement(positions):
    """Calculate total movement in recent frames"""
    if len(positions) < 2:
        return 0
    return np.sqrt((positions[-1][0]-positions[-2][0])**2 + 
                   (positions[-1][1]-positions[-2][1])**2)

def process_frame(frame, frame_count):
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
        
        # Update active frames
        if worker_tracks[track_id]['activity']:
            worker_tracks[track_id]['active_frames'] += 1
        worker_tracks[track_id]['total_frames'] = frame_count
        
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
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    processed_frame = process_frame(frame, frame_count)
    
    # Show results
    cv2.imshow('Factory Monitor', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate percentages and prepare results
results = []
for track_id, data in worker_tracks.items():
    percentage = (data['active_frames'] / data['total_frames']) * 100 if data['total_frames'] > 0 else 0
    results.append({
        'Worker ID': int(track_id),
        'Status': 'Active' if data['activity'] else 'Inactive',
        'Duration (frames)': data['total_frames'],
        'Percentage Active': f"{percentage:.2f}%"
    })

# Print results as a table
print("\nWorker Activity Summary:")
print("{:<10} {:<10} {:<15} {:<15}".format('Worker ID', 'Status', 'Duration (frames)', 'Percentage Active'))
for result in results:
    print("{:<10} {:<10} {:<15} {:<15}".format(result['Worker ID'], result['Status'], result['Duration (frames)'], result['Percentage Active']))

# Output results as an object
worker_activity_summary = {result['Worker ID']: result for result in results}
print("\nWorker Activity Summary Object:")
print(worker_activity_summary)
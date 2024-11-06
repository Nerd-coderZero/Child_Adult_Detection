import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
from tempfile import NamedTemporaryFile


# Initialize MediaPipe Pose for pose-based classification
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load the pre-trained TensorFlow Object Detection model
detection_model_dir = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
detection_model = tf.saved_model.load(detection_model_dir)

# Load the pre-trained child/adult classification model
child_adult_model = tf.keras.models.load_model('child_adult_model.h5')

def process_frame(frame, model):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections
    
def compute_iou(box1, box2):
    y_min1, x_min1, y_max1, x_max1 = box1
    y_min2, x_min2, y_max2, x_max2 = box2

    intersect_y_min = max(y_min1, y_min2)
    intersect_x_min = max(x_min1, x_min2)
    intersect_y_max = min(y_max1, y_max2)
    intersect_x_max = min(x_max1, x_max2)

    intersect_area = max(0, intersect_y_max - intersect_y_min) * max(0, intersect_x_max - intersect_x_min)
    box1_area = (y_max1 - y_min1) * (x_max1 - x_min1)
    box2_area = (y_max2 - y_min2) * (x_max2 - x_min2)

    union_area = box1_area + box2_area - intersect_area
    return intersect_area / union_area if union_area > 0 else 0

def classify_person_with_pose(frame, box, padding=0.1, child_threshold=0.8, adult_threshold=1.1):
    # Extract the bounding box coordinates
    y_min, x_min, y_max, x_max = box
    height, width, _ = frame.shape
    
    # Calculate padding (10% of bounding box size by default)
    pad_y = int((y_max - y_min) * padding * height)
    pad_x = int((x_max - x_min) * padding * width)
    
    # Ensure coordinates stay within frame boundaries
    start_point = (max(0, int(x_min * width - pad_x)), max(0, int(y_min * height - pad_y)))
    end_point = (min(width, int(x_max * width + pad_x)), min(height, int(y_max * height + pad_y)))
    
    # Crop the person from the frame using the adjusted bounding box
    cropped_person = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    
    # Convert the cropped image to RGB for MediaPipe Pose processing
    cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
    
    # Process the cropped image with MediaPipe Pose
    results = pose.process(cropped_rgb)
    
    if results.pose_landmarks:
        # Get the landmark positions for shoulder and ankle
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate average vertical distance from shoulders to ankles
        height_estimate = (abs(left_shoulder.y - left_ankle.y) + abs(right_shoulder.y - right_ankle.y)) / 2
        
        # Define thresholds for classification
        if height_estimate < child_threshold:
            return "Child"
        else: 
            return "Adult"
    
    # Fallback to the existing classification model if no pose landmarks are detected
    resized_person = cv2.resize(cropped_person, (128, 128))
    resized_person = resized_person / 255.0
    input_tensor = np.expand_dims(resized_person, axis=0)
    
    prediction = child_adult_model.predict(input_tensor)
    
    if prediction[0] < 0.6:
        return "Child"
    else:
        return "Adult"

def update_tracks(detections, tracks, iou_threshold=0.3, detection_threshold=0.7):
    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_scores = detections['detection_scores'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(int)
    
    person_detections = [(i, box) for i, (box, score, cls) in enumerate(zip(detection_boxes, detection_scores, detection_classes)) 
                         if cls == 1 and score > detection_threshold]
    
    updated_tracks = []
    used_detections = set()
    
    for track_id, track_box, track_age in tracks:
        best_iou = 0
        best_detection = None
        for detection_id, detection_box in person_detections:
            if detection_id in used_detections:
                continue
            iou = compute_iou(track_box, detection_box)
            if iou > best_iou:
                best_iou = iou
                best_detection = detection_id, detection_box
        
        if best_iou > iou_threshold:
            updated_tracks.append((track_id, best_detection[1], track_age + 1))
            used_detections.add(best_detection[0])
        else:
            if track_age > 0:
                updated_tracks.append((track_id, track_box, track_age - 1))
    
    max_id = max([id for id, _, _ in tracks]) if tracks else 0
    for detection_id, detection_box in person_detections:
        if detection_id not in used_detections:
            max_id += 1
            updated_tracks.append((max_id, detection_box, 1))
    
    return updated_tracks

def draw_boxes_and_ids(frame, tracks, min_age=3):
    for track_id, box, age in tracks:
        if age >= min_age:
            y_min, x_min, y_max, x_max = box
            start_point = (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]))
            end_point = (int(x_max * frame.shape[1]), int(y_max * frame.shape[0]))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            
            person_class = classify_person_with_pose(frame, box)
            
            cv2.putText(frame, f'{person_class} {track_id}', (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Streamlit app setup
st.title("Object Detection and Classification")
st.write("Upload a video file to detect and classify objects.")

# File upload
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    # Video processing
    cap = cv2.VideoCapture(temp_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Prepare an output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "processed_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracks = []
    frame_id = 0
    
    # Processing frames
    stframe = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 3 != 0:
            continue

        small_frame = cv2.resize(frame, (320, 240))
        detections = process_frame(small_frame, detection_model)
        tracks = update_tracks(detections, tracks)
        frame = draw_boxes_and_ids(frame, tracks)
        
        # Write frame to output video
        out.write(frame)

        # Update progress bar and display frame
        progress_bar.progress(min(frame_id / total_frames, 1.0))
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()
    
    st.success("Processing Complete")
    
    # Display output video
    st.video(output_path)

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
from tempfile import NamedTemporaryFile
import tempfile
import os

# Initialize MediaPipe Pose for pose-based classification
@st.cache_resource
def load_pose():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load models with caching
@st.cache_resource
def load_models():
    detection_model_dir = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
    detection_model = tf.saved_model.load(detection_model_dir)
    child_adult_model = tf.keras.models.load_model('child_adult_model.h5')
    return detection_model, child_adult_model

# Process frame with batch processing
@st.cache_data
def process_frame(frame, _model):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = _model(input_tensor)  # Use _model instead of model
    return detections


def process_video(uploaded_file, detection_model, pose, child_adult_model):
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        temp_path = os.path.join(temp_dir, "input.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output
        output_path = os.path.join(temp_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracking
        tracks = []
        frame_count = 0
        
        # Processing status
        status_text = st.empty()
        progress_bar = st.progress(0)
        preview = st.empty()
        
        # Process in batches
        batch_size = 5  # Process 5 frames at once
        while cap.isOpened():
            frames_batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)
                frame_count += 1
            
            if not frames_batch:
                break
                
            # Process batch
            for frame in frames_batch:
                # Resize for detection
                small_frame = cv2.resize(frame, (320, 240))
                detections = process_frame(small_frame, detection_model)
                
                # Update tracking and draw
                tracks = update_tracks(detections, tracks)
                processed_frame = draw_boxes_and_ids(frame, tracks, pose, child_adult_model)
                
                # Write and update UI
                out.write(processed_frame)
                if frame_count % 10 == 0:  # Update preview less frequently
                    preview.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Read the output video for display
        with open(output_path, "rb") as f:
            return f.read()
            
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

def draw_boxes_and_ids(frame, tracks, pose, child_adult_model, min_age=3):
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


def main():
    st.title("Person Detection and Classification")
    
    # Load models
    try:
        detection_model, child_adult_model = load_models()
        pose = load_pose()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
        
    # File upload with size limit
    uploaded_file = st.file_uploader(
        "Choose a video file (max 200MB)",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
            st.error("File too large. Please upload a video smaller than 200MB.")
            return
            
        try:
            with st.spinner("Processing video..."):
                processed_video = process_video(uploaded_file, detection_model, pose, child_adult_model)
                
            # Display processed video
            if processed_video:
                st.success("Processing complete!")
                st.video(processed_video)
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()

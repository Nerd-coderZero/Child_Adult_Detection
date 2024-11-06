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
    return mp_pose, mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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
    detections = _model(input_tensor)
    return detections

def classify_person_with_pose(frame, box, pose_model, mp_pose, child_adult_model, padding=0.1, child_threshold=0.8, adult_threshold=1.1):
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
    
    if cropped_person.size == 0:  # Check if the cropped image is empty
        return "Unknown"
    
    # Convert the cropped image to RGB for MediaPipe Pose processing
    cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
    
    # Process the cropped image with MediaPipe Pose
    results = pose_model.process(cropped_rgb)
    
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
    try:
        resized_person = cv2.resize(cropped_person, (128, 128))
        resized_person = resized_person / 255.0
        input_tensor = np.expand_dims(resized_person, axis=0)
        
        prediction = child_adult_model.predict(input_tensor)
        
        if prediction[0] < 0.6:
            return "Child"
        else:
            return "Adult"
    except Exception:
        return "Unknown"

def draw_boxes_and_ids(frame, tracks, pose_model, mp_pose, child_adult_model, min_age=3):
    processed_frame = frame.copy()
    for track_id, box, age in tracks:
        if age >= min_age:
            y_min, x_min, y_max, x_max = box
            start_point = (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]))
            end_point = (int(x_max * frame.shape[1]), int(y_max * frame.shape[0]))
            cv2.rectangle(processed_frame, start_point, end_point, (0, 255, 0), 2)
            
            person_class = classify_person_with_pose(frame, box, pose_model, mp_pose, child_adult_model)
            
            cv2.putText(processed_frame, f'{person_class} {track_id}', (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return processed_frame

def process_video(uploaded_file, detection_model, pose_model, mp_pose, child_adult_model):
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
        
        try:
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
                    processed_frame = draw_boxes_and_ids(frame, tracks, pose_model, mp_pose, child_adult_model)
                    
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
                
        except Exception as e:
            cap.release()
            out.release()
            raise e

def main():
    st.title("Person Detection and Classification")
    
    # Load models
    try:
        detection_model, child_adult_model = load_models()
        mp_pose, pose_model = load_pose()
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
                processed_video = process_video(
                    uploaded_file, 
                    detection_model, 
                    pose_model,
                    mp_pose,
                    child_adult_model
                )
                
            # Display processed video
            if processed_video:
                st.success("Processing complete!")
                st.video(processed_video)
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()

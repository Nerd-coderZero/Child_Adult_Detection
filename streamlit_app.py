import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
from tempfile import NamedTemporaryFile, gettempdir
import logging
from typing import Generator
import gc
import pathlib
import urllib.request
import shutil
import os
import urllib.request
from pathlib import Path





# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.model_dir = Path("./mediapipe_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Define MediaPipe model paths
        os.environ["MEDIAPIPE_POSE_LANDMARK_LITE"] = str(self.model_dir / "pose_landmark_lite.tflite")
        os.environ["MEDIAPIPE_POSE_LANDMARK_FULL"] = str(self.model_dir / "pose_landmark_full.tflite")

        # Download model files if they don't already exist
        if not (self.model_dir / "pose_landmark_lite.tflite").exists():
            self.download_model("https://storage.googleapis.com/mediapipe-models/pose_landmark_lite.tflite", 
                                self.model_dir / "pose_landmark_lite.tflite")

        if not (self.model_dir / "pose_landmark_full.tflite").exists():
            self.download_model("https://storage.googleapis.com/mediapipe-models/pose_landmark_full.tflite", 
                                self.model_dir / "pose_landmark_full.tflite")

        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def download_model(self, url, destination):
        """Download model file if not already present."""
        urllib.request.urlretrieve(url, destination)

    def process_video(self, video_path: str, progress_bar, stframe, max_frames: int = 300):
        """Process video with memory-efficient streaming."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // max_frames)
        processed_frames = 0

        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if processed_frames % frame_skip == 0:
                # Process frame
                processed_frame = self._process_single_frame(frame)

                # Update progress and display frame
                progress_bar.progress(processed_frames / min(total_frames, max_frames))
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {processed_frames}", use_column_width=True)

                del processed_frame  # Force garbage collection
                gc.collect()

            processed_frames += 1

        cap.release()

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        # Run MediaPipe Pose processing here (adjust this if additional processing is needed)
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        return frame

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
    
    
    
    def load_models(self):
        """Lazy loading of models with error handling"""
        if self.detection_model is None:
            try:
                # Check if model files exist
                model_path = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
                if not os.path.exists(model_path):
                    logger.error(f"Detection model not found at {model_path}")
                else:
                    logger.info("Detection model path verified.")
                    
                self.detection_model = tf.saved_model.load(model_path)
                logger.info("Detection model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading detection model: {e}")
                st.error("Error loading detection model. Please check if model files are present.")
                raise
                
        if self.child_adult_model is None:
            try:
                # Check if model file exists
                if not os.path.exists('child_adult_model.h5'):
                    raise FileNotFoundError("Classification model not found")
                
                self.child_adult_model = tf.keras.models.load_model('child_adult_model.h5', compile=False)
                self.child_adult_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Classification model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading classification model: {e}")
                st.error("Error loading classification model. Please check if model file is present.")
                raise

    

   

    

    def _update_tracks(self, detections: dict, tracks: list) -> list:
        """Update object tracks with memory optimization"""
        # Extract relevant detection data
        boxes = detections['detection_boxes'].numpy()[0]
        scores = detections['detection_scores'].numpy()[0]
        classes = detections['detection_classes'].numpy()[0]
        
        # Filter for people with good confidence
        valid_detections = [
            (i, box) for i, (box, score, cls) in 
            enumerate(zip(boxes, scores, classes))
            if cls == 1 and score > 0.5
        ]
        
        # Update existing tracks
        updated_tracks = []
        used_detections = set()
        
        for track in tracks:
            if len(updated_tracks) >= 10:  # Limit number of tracked objects
                break
                
            track_id, box, age = track
            best_match = None
            best_iou = 0.3  # IOU threshold
            
            for det_id, det_box in valid_detections:
                if det_id not in used_detections:
                    iou = self._compute_iou(box, det_box)
                    if iou > best_iou:
                        best_match = (det_id, det_box)
                        best_iou = iou
            
            if best_match:
                updated_tracks.append((track_id, best_match[1], age + 1))
                used_detections.add(best_match[0])
            elif age > 0:
                updated_tracks.append((track_id, box, age - 1))
        
        # Add new tracks (limited)
        max_id = max([id for id, _, _ in tracks], default=0)
        for det_id, det_box in valid_detections:
            if det_id not in used_detections and len(updated_tracks) < 10:
                max_id += 1
                updated_tracks.append((max_id, det_box, 1))
        
        return updated_tracks

    def _draw_results(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """Draw detection results with basic classification"""
        result = frame.copy()
        for track_id, box, age in tracks:
            if age < 3:  # Skip unstable tracks
                continue
                
            # Convert box coordinates
            h, w = frame.shape[:2]
            y1, x1, y2, x2 = [int(coord * dim) for coord, dim in zip(box, [h, w, h, w])]
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Simple classification based on box height
            person_type = "Child" if (y2 - y1) < (h * 0.5) else "Adult"
            
            # Draw label
            label = f"{person_type} {track_id}"
            cv2.putText(result, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result

    @staticmethod
    def _compute_iou(box1, box2):
        """Compute Intersection over Union"""
        y1, x1, y3, x3 = box1
        y2, x2, y4, x4 = box2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x3, x4), min(y3, y4)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x3 - x1) * (y3 - y1)
        box2_area = (x4 - x2) * (y4 - y2)
        
        return intersection / (box1_area + box2_area - intersection)

def main():
    st.set_page_config(
        page_title="Video Analysis",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Person Detection and Classification")
    st.write("Upload a video file (max 50MB) for analysis")
    
    # Add a warning about GPU
    st.warning("Note: This application is running on CPU. Some operations may be slower than on GPU.")
    
    # File uploader with size limit
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )
    
    if uploaded_file:
        # Check file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        if file_size > 50:
            st.error("File too large. Please upload a video smaller than 50MB")
            return
            
        processor = None
        try:
            # Save file temporarily
            with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
            
            # Initialize processor and UI elements
            processor = VideoProcessor()
            progress_bar = st.progress(0)
            stframe = st.empty()
            
            # Process video
            processor.process_video(video_path, progress_bar, stframe)
            
            # Cleanup
            os.unlink(video_path)
            st.success("Processing complete!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)
        
        finally:
            # Clean up resources
            if processor:
                processor.cleanup()
            gc.collect()

if __name__ == "__main__":
    main()

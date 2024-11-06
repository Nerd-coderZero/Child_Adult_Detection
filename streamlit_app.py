import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import logging
from typing import Generator, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class VideoProcessor:
    def __init__(self, detection_model_path: str, classification_model_path: str):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        # Load models
        try:
            self.detection_model = tf.saved_model.load(detection_model_path)
            self.child_adult_model = tf.keras.models.load_model(classification_model_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def process_video_generator(self, video_path: str, chunk_size: int = 30) -> Generator[np.ndarray, None, None]:
        """Process video in chunks to manage memory usage."""
        cap = cv2.VideoCapture(video_path)
        tracks = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                frames_processed = 0
                while frames_processed < chunk_size:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    
                    frame_count += 1
                    if frame_count % 3 != 0:  # Process every third frame
                        continue
                        
                    processed_frame = self._process_single_frame(frame, tracks)
                    yield processed_frame
                    frames_processed += 1
                    
        finally:
            cap.release()

    def _process_single_frame(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """Process a single frame with detection and tracking."""
        small_frame = cv2.resize(frame, (320, 240))
        detections = self._detect_objects(small_frame)
        tracks = self._update_tracks(detections, tracks)
        return self._draw_boxes_and_ids(frame, tracks)

    def _detect_objects(self, frame: np.ndarray) -> dict:
        """Perform object detection on a frame."""
        input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]
        return self.detection_model(input_tensor)

    @staticmethod
    def _compute_iou(box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union for two boxes."""
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

    def _classify_person(self, frame: np.ndarray, box: Tuple[float, float, float, float]) -> str:
        """Classify a person as child or adult using pose estimation."""
        y_min, x_min, y_max, x_max = box
        height, width = frame.shape[:2]
        
        # Extract person from frame
        person_frame = frame[
            max(0, int(y_min * height)):min(height, int(y_max * height)),
            max(0, int(x_min * width)):min(width, int(x_max * width))
        ]
        
        if person_frame.size == 0:
            return "Unknown"

        # Use pose estimation for classification
        try:
            results = self.pose.process(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                height_estimate = self._calculate_pose_height(results)
                return "Child" if height_estimate < 0.8 else "Adult"
        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")

        # Fallback to ML classification
        try:
            resized = cv2.resize(person_frame, (128, 128)) / 255.0
            prediction = self.child_adult_model.predict(np.expand_dims(resized, axis=0), verbose=0)
            return "Child" if prediction[0] < 0.6 else "Adult"
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "Unknown"

    def _calculate_pose_height(self, pose_results) -> float:
        """Calculate relative height from pose landmarks."""
        landmarks = pose_results.pose_landmarks.landmark
        left_height = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y - 
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y)
        right_height = abs(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y -
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y)
        return (left_height + right_height) / 2

def main():
    st.set_page_config(page_title="Video Analysis", layout="wide")
    st.title("Person Detection and Classification")
    
    # Initialize processor
    try:
        processor = VideoProcessor(
            "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model",
            "child_adult_model.h5"
        )
    except Exception as e:
        st.error(f"Failed to initialize models: {e}")
        return

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        try:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Process video in chunks
            stframe = st.empty()
            for processed_frame in processor.process_video_generator(video_path):
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                            caption="Processed video", use_column_width=True)
            
            # Cleanup
            os.unlink(video_path)
            st.success("Video processing completed!")
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            logger.error(f"Processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

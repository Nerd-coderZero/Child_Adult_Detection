# streamlit_app.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
from tempfile import NamedTemporaryFile
import os
import logging
from typing import Generator, Tuple

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        # Load models
        try:
            self.detection_model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")
            self.child_adult_model = tf.keras.models.load_model('child_adult_model.h5')
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def process_video_generator(self, video_path: str, chunk_size: int = 30) -> Generator[np.ndarray, None, None]:
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
        try:
            small_frame = cv2.resize(frame, (320, 240))
            detections = self._detect_objects(small_frame)
            tracks = self._update_tracks(detections, tracks)
            return self._draw_boxes_and_ids(frame, tracks)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def _detect_objects(self, frame: np.ndarray) -> dict:
        input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]
        return self.detection_model(input_tensor)

    def _update_tracks(self, detections: dict, tracks: list, 
                      iou_threshold: float = 0.3, 
                      detection_threshold: float = 0.7) -> list:
        detection_boxes = detections['detection_boxes'].numpy()[0]
        detection_scores = detections['detection_scores'].numpy()[0]
        detection_classes = detections['detection_classes'].numpy()[0].astype(int)
        
        person_detections = [
            (i, box) for i, (box, score, cls) in 
            enumerate(zip(detection_boxes, detection_scores, detection_classes))
            if cls == 1 and score > detection_threshold
        ]
        
        updated_tracks = []
        used_detections = set()
        
        # Update existing tracks
        for track_id, track_box, track_age in tracks:
            best_match = max(
                ((det_id, det_box, self._compute_iou(track_box, det_box))
                 for det_id, det_box in person_detections
                 if det_id not in used_detections),
                key=lambda x: x[2],
                default=(None, None, 0)
            )
            
            if best_match[2] > iou_threshold:
                updated_tracks.append((track_id, best_match[1], track_age + 1))
                used_detections.add(best_match[0])
            elif track_age > 0:
                updated_tracks.append((track_id, track_box, track_age - 1))
        
        # Add new tracks
        max_id = max([id for id, _, _ in tracks], default=0)
        for det_id, det_box in person_detections:
            if det_id not in used_detections:
                max_id += 1
                updated_tracks.append((max_id, det_box, 1))
        
        return updated_tracks

    def _draw_boxes_and_ids(self, frame: np.ndarray, tracks: list, min_age: int = 3) -> np.ndarray:
        annotated_frame = frame.copy()
        for track_id, box, age in tracks:
            if age >= min_age:
                y_min, x_min, y_max, x_max = box
                start_point = (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]))
                end_point = (int(x_max * frame.shape[1]), int(y_max * frame.shape[0]))
                
                # Classify person
                person_class = self._classify_person(frame, box)
                
                # Draw box and label
                cv2.rectangle(annotated_frame, start_point, end_point, (0, 255, 0), 2)
                label = f'{person_class} {track_id}'
                cv2.putText(
                    annotated_frame, label,
                    (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )
        
        return annotated_frame

    def _classify_person(self, frame: np.ndarray, box: Tuple[float, float, float, float]) -> str:
        try:
            y_min, x_min, y_max, x_max = box
            height, width = frame.shape[:2]
            
            # Extract person from frame
            person_frame = frame[
                max(0, int(y_min * height)):min(height, int(y_max * height)),
                max(0, int(x_min * width)):min(width, int(x_max * width))
            ]
            
            if person_frame.size == 0:
                return "Unknown"

            # Try pose estimation first
            rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                height_estimate = self._calculate_pose_height(pose_results)
                if height_estimate is not None:
                    return "Child" if height_estimate < 0.8 else "Adult"
            
            # Fallback to ML classification
            resized = cv2.resize(person_frame, (128, 128)) / 255.0
            prediction = self.child_adult_model.predict(
                np.expand_dims(resized, axis=0),
                verbose=0
            )
            return "Child" if prediction[0] < 0.6 else "Adult"
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "Unknown"

    def _calculate_pose_height(self, pose_results) -> float:
        try:
            landmarks = pose_results.pose_landmarks.landmark
            left_height = abs(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y -
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
            )
            right_height = abs(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y -
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
            )
            return (left_height + right_height) / 2
        except:
            return None

    @staticmethod
    def _compute_iou(box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float]) -> float:
        y_min1, x_min1, y_max1, x_max1 = box1
        y_min2, x_min2, y_max2, x_max2 = box2

        intersect_y_min = max(y_min1, y_min2)
        intersect_x_min = max(x_min1, x_min2)
        intersect_y_max = min(y_max1, y_max2)
        intersect_x_max = min(x_max1, x_max2)

        intersect_area = max(0, intersect_y_max - intersect_y_min) * \
                        max(0, intersect_x_max - intersect_x_min)
        
        box1_area = (y_max1 - y_min1) * (x_max1 - x_min1)
        box2_area = (y_max2 - y_min2) * (x_max2 - x_min2)
        
        union_area = box1_area + box2_area - intersect_area
        return intersect_area / union_area if union_area > 0 else 0

def main():
    st.set_page_config(page_title="Video Analysis", layout="wide")
    st.title("Person Detection and Classification")
    
    # Initialize processor
    try:
        processor = VideoProcessor()
    except Exception as e:
        st.error(f"Failed to initialize models: {e}")
        return

    # Main content
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        try:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Add file size check
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Convert to MB
            if file_size > 100:  # Limit to 100MB
                st.error("File size too large. Please upload a video smaller than 100MB.")
                os.unlink(video_path)
                return

            # Process video in chunks
            progress_bar = st.progress(0)
            stframe = st.empty()
            
            # Get total frames for progress calculation
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Process frames
            frame_count = 0
            for processed_frame in processor.process_video_generator(video_path):
                frame_count += 3  # Since we're processing every third frame
                progress = min(1.0, frame_count / total_frames)
                progress_bar.progress(progress)
                
                stframe.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption="Processed video",
                    use_column_width=True
                )
            
            # Cleanup
            os.unlink(video_path)
            st.success("Video processing completed!")
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            logger.error(f"Processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
from typing import Tuple, List, Dict, Optional
import tempfile
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import os
import sys
import gdown
import subprocess


# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPersonTracker:
    def __init__(
            self,
            model_path: str = 'best_model.pth',
            confidence_threshold: float = 0.75,  # Increased confidence threshold
            input_size: tuple = (416, 416),
            close_up_min_pixels: int = 60,  # Increased minimum size
            close_up_max_pixels: int = 250,  # Reduced maximum size
            close_up_aspect_ratio_bounds: Tuple[float, float] = (0.4, 1.8),  # Tightened bounds
            boundary_margin_ratio: float = 0.15,  # Increased margin
            buffer_size: int = 5  # Added frame buffer size
        ):
            # Setup logging and device
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            self.input_size = input_size
            self.buffer_size = buffer_size
            self.frame_buffer = []

            # Optimized parameters for false positive reduction
            self.detection_params = {
                'max_size_ratio': 0.80,  # Reduced to avoid large false detections
                'min_pixels': 80,  # Increased minimum size
                'aspect_ratio_bounds': (0.3, 2.0),  # Tightened aspect ratio bounds
                'padding_ratio': 0.05,  # Reduced padding
                'temporal_smoothing': 7,  # Increased temporal smoothing
                'min_detection_confidence': 0.6,  # Minimum detection confidence
                'track_stability_threshold': 5  # Minimum frames for stable track
            }

            # Enhanced tracking parameters
            self.tracking_params = {
                'velocity_persistence': 0.7,  # How much to consider previous velocity
                'position_smoothing': 0.8,  # Position smoothing factor
                'min_hits': 3,  # Minimum hits before establishing track
                'max_age': 20,  # Maximum frames to keep lost tracks
                'min_iou': 0.3  # Minimum IOU for track association
            }

            self.close_up_min_pixels = close_up_min_pixels
            self.close_up_max_pixels = close_up_max_pixels
            self.close_up_aspect_ratio_bounds = close_up_aspect_ratio_bounds
            self.confidence_threshold = confidence_threshold
            self.boundary_margin_ratio = boundary_margin_ratio

            # Enhanced classification history with confidence weighting
            self.classification_history = {}
            self.track_stability = {}

            try:
                self.classification_model = self._load_model(model_path)
                self.detector = self._load_detector()
                self._initialize_tracker()
            except Exception as e:
                self.logger.error(f"Error initializing models: {e}")
                raise

    def _initialize_tracker(self):
        """Initialize DeepSort tracker with parameters optimized for close interaction"""
    
        self.tracker = DeepSort(
            max_age=self.tracking_params['max_age'],
            n_init=self.tracking_params['min_hits'],
            max_iou_distance=1.0 - self.tracking_params['min_iou'],
            max_cosine_distance=0.25,
            nn_budget=100,
            override_track_class=None,
            embedder=self.embedder,  # Add the embedder here
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available()
        )

    def _load_model(self, model_path: str) -> nn.Module:
        """Load classification model with proper error handling"""
        model = models.efficientnet_b0(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 2)
        )

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")

    def _update_track_stability(self, track_id: int, detection_confidence: float):
        """Update track stability metrics"""
        if track_id not in self.track_stability:
            self.track_stability[track_id] = {
                'consecutive_detections': 0,
                'total_detections': 0,
                'average_confidence': 0
            }

        stability = self.track_stability[track_id]
        stability['consecutive_detections'] += 1
        stability['total_detections'] += 1
        stability['average_confidence'] = (
            stability['average_confidence'] * (stability['total_detections'] - 1) +
            detection_confidence
        ) / stability['total_detections']

    def _is_track_stable(self, track_id: int) -> bool:
        """Check if a track is stable enough for classification"""
        if track_id not in self.track_stability:
            return False

        stability = self.track_stability[track_id]
        return (
            stability['consecutive_detections'] >= self.detection_params['track_stability_threshold'] and
            stability['average_confidence'] >= self.detection_params['min_detection_confidence']
        )
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection with fixed color space conversion"""
        try:
            # Optional: Resize if frame is too large
            max_dim = 1280
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # Split channels
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)

            # Merge channels back
            enhanced_lab = cv2.merge([l, a, b])

            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced_bgr

        except Exception as e:
            self.logger.warning(f"Error in preprocessing: {e}")
            return frame  # Return original frame if preprocessing fails

    def _load_detector(self) -> any:
        """Load YOLOv8 detector with optimized parameters"""
        try:
            from ultralytics import YOLO
            detector = YOLO('yolov8n.pt')
            detector.to(self.device)
            # Set inference size to match training size
            detector.overrides['imgsz'] = self.input_size
            return detector
        except Exception as e:
            self.logger.error(f"Error loading YOLOv8 detector: {e}")
            raise

    def detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in frame with YOLOv8"""
        # Preprocess frame
        processed_frame = self._preprocess_frame(frame)

        # Run detection
        results = self.detector(processed_frame, classes=[0])  # 0 is person class
        detections = []

        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Get box coordinates
                conf = float(box.conf)  # Get confidence score

                if self._validate_detection(bbox, frame.shape):
                    padded_bbox = self._add_padding(bbox, frame.shape)
                    detection = np.array([*padded_bbox, conf])
                    detections.append(detection)

        return np.array(detections) if detections else np.empty((0, 5))

    def _validate_detection(self, bbox: np.ndarray, frame_shape: tuple) -> bool:
        """Enhanced validation with additional checks for close objects"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        frame_height, frame_width = frame_shape[:2]

        # Basic size validation
        if (width < self.detection_params['min_pixels'] or 
            height < self.detection_params['min_pixels'] or
            width > frame_width * self.detection_params['max_size_ratio'] or
            height > frame_height * self.detection_params['max_size_ratio']):
            return False

        # Aspect ratio validation
        aspect_ratio = width / height
        min_ratio, max_ratio = self.detection_params['aspect_ratio_bounds']
        if not min_ratio <= aspect_ratio <= max_ratio:
            # Check for close-up objects
            if self.close_up_min_pixels <= width * height <= self.close_up_max_pixels:
                min_ratio, max_ratio = self.close_up_aspect_ratio_bounds
                if min_ratio <= aspect_ratio <= max_ratio:
                    return True
            return False

        # Enhanced boundary detection validation
        margin_x = int(frame_width * self.boundary_margin_ratio)
        margin_y = int(frame_height * self.boundary_margin_ratio)
        if (x1 < margin_x or y1 < margin_y or 
            x2 > frame_width - margin_x or y2 > frame_height - margin_y):
            # Additional size check for boundary objects
            if width * height > (frame_width * frame_height * 0.9):  # If object is too large
                return False

        return True

    def _get_consistent_detections(self, frame: np.ndarray) -> np.ndarray:
        """Get detections with temporal consistency check"""
        current_detections = self.detect_persons(frame)
        
        if len(self.frame_buffer) < 2:
            return current_detections

        # Compare with previous frame detections
        prev_detections = self.detect_persons(self.frame_buffer[-2])
        
        consistent_detections = []
        for det in current_detections:
            # Check if detection has corresponding detection in previous frame
            if self._has_matching_detection(det, prev_detections):
                consistent_detections.append(det)

        return np.array(consistent_detections)

    def _has_matching_detection(self, detection: np.ndarray, prev_detections: np.ndarray,
                              iou_threshold: float = 0.3) -> bool:
        """Check if detection has a match in previous frame"""
        if len(prev_detections) == 0:
            return False

        # Calculate IoU with all previous detections
        ious = np.array([self._calculate_iou(detection[:4], prev_det[:4]) 
                        for prev_det in prev_detections])
        
        return np.any(ious > iou_threshold)

    def _has_significant_overlap(self, bbox: np.ndarray, tracks: List, track_id: int,
                               overlap_threshold: float = 0.3) -> bool:
        """Check for significant overlap with other tracks"""
        for other_track in tracks:
            if other_track.track_id != track_id and other_track.is_confirmed():
                other_ltwh = other_track.to_ltwh()
                other_bbox = np.array([
                    other_ltwh[0], other_ltwh[1],
                    other_ltwh[0] + other_ltwh[2], other_ltwh[1] + other_ltwh[3]
                ])
                
                if self._calculate_iou(bbox, other_bbox) > overlap_threshold:
                    return True
        return False
                                   
    def _smooth_classification(self, track_id: int, label: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to classifications"""
        history = self.classification_history.get(track_id, [])
        history.append((label, confidence))

        # Keep only recent history
        history = history[-self.detection_params['temporal_smoothing']:]
        self.classification_history[track_id] = history

        # Count occurrences and average confidence for each label
        label_stats = {}
        for l, c in history:
            if l not in label_stats:
                label_stats[l] = {'count': 0, 'confidence_sum': 0}
            label_stats[l]['count'] += 1
            label_stats[l]['confidence_sum'] += c

        # Find most common label
        max_count = 0
        smoothed_label = None
        smoothed_confidence = 0

        for l, stats in label_stats.items():
            if stats['count'] > max_count:
                max_count = stats['count']
                smoothed_label = l
                smoothed_confidence = stats['confidence_sum'] / stats['count']

        return smoothed_label, smoothed_confidence

    def _add_padding(self, bbox: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """Add padding to detection bbox"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        padding = self.detection_params['padding_ratio']

        x1 = max(0, x1 - width * padding)
        x2 = min(frame_shape[1], x2 + width * padding)
        y1 = max(0, y1 - height * padding)
        y2 = min(frame_shape[0], y2 + height * padding)

        return np.array([x1, y1, x2, y2])

    def predict_person(self, frame: np.ndarray, bbox: np.ndarray, track_id: int) -> Tuple[str, float]:
        """Enhanced classification with proper resizing"""
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            person_img = frame[y1:y2, x1:x2]

            if person_img.size == 0:
                return None, 0

            # Maintain aspect ratio while resizing to training dimensions
            h, w = person_img.shape[:2]
            target_h, target_w = self.input_size

            # Calculate scaling factor while maintaining aspect ratio
            scale = min(target_w/w, target_h/h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize maintaining aspect ratio
            person_img = cv2.resize(person_img, (new_w, new_h))

            # Create a blank canvas of target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Calculate position to paste the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2

            # Paste the resized image onto the canvas
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = person_img

            # Convert to RGB and normalize
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas = canvas / 255.0

            # Convert to tensor
            person_img = torch.FloatTensor(canvas).permute(2, 0, 1).unsqueeze(0)
            person_img = person_img.to(self.device)

            with torch.no_grad():
                outputs = self.classification_model(person_img)
                probs = torch.softmax(outputs, dim=1)

                child_prob = float(probs[0, 1])
                therapist_prob = float(probs[0, 0])

                if child_prob > self.confidence_threshold:
                    label, confidence = 'Child', child_prob
                elif therapist_prob > self.confidence_threshold:
                    label, confidence = 'Therapist', therapist_prob
                else:
                    label = 'Child' if child_prob > therapist_prob else 'Therapist'
                    confidence = max(child_prob, therapist_prob)

                return self._smooth_classification(track_id, label, confidence)

        except Exception as e:
            self.logger.error(f"Error in classification: {e}")
            return None, 0

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_display: bool = True):
        """Process video with enhanced tracking and stability checks"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video file {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Update frame buffer for temporal consistency
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)

                # Get detections with temporal consistency check
                detections = self._get_consistent_detections(frame)
                
                if len(detections) > 0:
                    # Convert detections to DeepSort format
                    detection_list = []
                    for det in detections:
                        detection_list.append(([det[0], det[1], det[2] - det[0], det[3] - det[1]], det[4], 'person'))

                    # Update tracks with enhanced stability checking
                    tracks = self.tracker.update_tracks(detection_list, frame=frame)

                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        ltwh = track.to_ltwh()
                        bbox = np.array([
                            ltwh[0], ltwh[1],
                            ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]
                        ])

                        # Update track stability
                        self._update_track_stability(track_id, track.get_det_conf())

                        # Only process stable tracks
                        if self._is_track_stable(track_id):
                            # Check for significant overlap with other tracks
                            if not self._has_significant_overlap(bbox, tracks, track_id):
                                label, confidence = self.predict_person(frame, bbox, track_id)
                                if label is not None:
                                    self.draw_detection(frame, bbox, track_id, label, confidence)

                if show_display:
                    cv2.imshow('Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if output_path:
                    out.write(frame)

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
        finally:
            cap.release()
            if output_path:
                out.release()
            if show_display:
                cv2.destroyAllWindows()

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection coordinates
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def draw_detection(self, frame: np.ndarray, bbox: np.ndarray, track_id: int, 
                      label: str, confidence: float):
        """Draw detection box and label"""
        x1, y1, x2, y2 = map(int, bbox)

        # Color based on class
        color = (0, 255, 0) if label == 'Therapist' else (0, 165, 255)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{label} ({confidence:.2f}) ID:{track_id}"
        label_width, label_height = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Adjust label position to stay at the top of the detection box
        label_x = max(x1, 0)
        label_y = max(y1 - 10, 0)

        # Draw the label background with padding
        cv2.rectangle(frame, (label_x, label_y), (label_x + label_width, label_y + label_height), color, -1)

        # Draw the label text
        cv2.putText(frame, label_text, (label_x, label_y + label_height - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def load_model():
    """Initialize the tracker with proper error handling for Streamlit"""
    try:
        if st.session_state.tracker is None:
            st.session_state.tracker = EnhancedPersonTracker(
                model_path='best_model.pth',
                confidence_threshold=0.6
            )
        return st.session_state.tracker
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_uploaded_video(video_file):
    """Process uploaded video file"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, 'input.mp4')
        output_path = os.path.join(temp_dir, 'output.mp4')

        # Save uploaded file
        with open(input_path, 'wb') as f:
            f.write(video_file.read())
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            detections = st.session_state.tracker.detect_persons(frame)
            if len(detections) > 0:
                detection_list = []
                for det in detections:
                    detection_list.append(([det[0], det[1], det[2] - det[0], det[3] - det[1]], det[4], 'person'))
                
                tracks = st.session_state.tracker.tracker.update_tracks(detection_list, frame=frame)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    track_id = track.track_id
                    ltwh = track.to_ltwh()
                    bbox = np.array([
                        ltwh[0], ltwh[1],
                        ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]
                    ])
                    
                    label, confidence = st.session_state.tracker.predict_person(frame, bbox, track_id)
                    if label is not None:
                        st.session_state.tracker.draw_detection(frame, bbox, track_id, label, confidence)

            out.write(frame)
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            progress_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Clean up
        cap.release()
        out.release()
        
        return output_path

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None
    finally:
        # Clean up temp files
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

class VideoProcessor:
    def __init__(self):
        self.tracker = st.session_state.tracker

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Process frame
            detections = self.tracker.detect_persons(img)
            if len(detections) > 0:
                detection_list = []
                for det in detections:
                    detection_list.append(([det[0], det[1], det[2] - det[0], det[3] - det[1]], det[4], 'person'))
                
                tracks = self.tracker.tracker.update_tracks(detection_list, frame=img)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    track_id = track.track_id
                    ltwh = track.to_ltwh()
                    bbox = np.array([
                        ltwh[0], ltwh[1],
                        ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]
                    ])
                    
                    label, confidence = self.tracker.predict_person(img, bbox, track_id)
                    if label is not None:
                        self.tracker.draw_detection(img, bbox, track_id, label, confidence)
        
        except Exception as e:
            st.error(f"Error processing webcam frame: {str(e)}")
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Person Tracking and Classification App")
    
    # Initialize model first
    with st.spinner("Loading model..."):
        tracker = load_model()
        if tracker is None:
            st.error("Failed to initialize the model. Please check if model file exists.")
            return

    # Sidebar for app options
    st.sidebar.title("Settings")
    input_option = st.sidebar.radio(
        "Choose Input Source",
        ["Upload Video", "Use Webcam"]
    )
    
    if input_option == "Upload Video":
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_path = process_uploaded_video(uploaded_file)
                    
                    if output_path and os.path.exists(output_path):
                        st.success("Video processed successfully!")
                        
                        # Save to session state
                        with open(output_path, 'rb') as file:
                            st.session_state.processed_video = file.read()
                        
                        # Display video
                        st.video(output_path)
                        
                        # Download button
                        st.download_button(
                            label="Download processed video",
                            data=st.session_state.processed_video,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
    
    else:  # Webcam option
        st.write("Webcam Feed(Does not work yet please upload video file.)")
        
        
        webrtc_ctx = webrtc_streamer(
            key="person-tracking",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

if __name__ == "__main__":
    main()

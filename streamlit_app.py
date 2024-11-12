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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import queue
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global queue for frame processing
FRAME_QUEUE = queue.Queue()
RESULT_QUEUE = queue.Queue()
class EnhancedPersonTracker:
    # ... [Keep the existing EnhancedPersonTracker class implementation] ...
    # Note: The original class remains unchanged, just copy it here
    def __init__(
        self,
        model_path: str = 'best_model.pth',
        confidence_threshold: float = 0.6,
        close_up_min_pixels: int = 60,
        close_up_max_pixels: int = 300,
        close_up_aspect_ratio_bounds: Tuple[float, float] = (0.5, 2.0),
        boundary_margin_ratio: float = 0.1
    ):
        # Setup logging and device
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Core parameters - adjusted for better detection
        self.detection_params = {
            'max_size_ratio': 0.95,  # Increased to handle close objects
            'min_pixels': 40,  # Increased minimum size to reduce false positives
            'aspect_ratio_bounds': (0.3, 3.0),  # Tightened aspect ratio bounds
            'padding_ratio': 0.08,  # Reduced padding to prevent overlap
            'temporal_smoothing': 5  # Frames for temporal smoothing
        }
        
        # Additional parameters for close-up handling
        self.close_up_min_pixels = close_up_min_pixels
        self.close_up_max_pixels = close_up_max_pixels
        self.close_up_aspect_ratio_bounds = close_up_aspect_ratio_bounds
        self.confidence_threshold = confidence_threshold
        self.boundary_margin_ratio = boundary_margin_ratio
        
        # Initialize classification history
        self.classification_history = {}
        
        # Initialize models as before
        try:
            self.classification_model = self._load_model(model_path)
            self.detector = self._load_detector()
            self._initialize_tracker()
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    
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
            
    def load_model():
        """Initialize the tracker with proper error handling for Streamlit"""
        try:
            tracker = EnhancedPersonTracker(
                model_path='best_model.pth',
                confidence_threshold=0.6
            )
            return tracker
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def _initialize_tracker(self):
        """Initialize DeepSort tracker with parameters optimized for close interaction"""
        self.tracker = DeepSort(
            max_age=30,            # Reduced to prevent tracking ghost objects
            n_init=3,              # Increased to ensure more stable initialization
            max_iou_distance=0.6,  # Reduced for better discrimination
            max_cosine_distance=0.3,  # Reduced for more strict appearance matching
            nn_budget=100
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
            detector = YOLO('yolov8n.pt')  # Using nano model for speed, use 's' or 'm' for better accuracy
            detector.to(self.device)
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
        """Enhanced classification with temporal smoothing"""
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size == 0:
                return None, 0
                
            # Prepare image with enhanced preprocessing
            person_img = cv2.resize(person_img, (224, 224))
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            
            # Add contrast enhancement
            lab = cv2.cvtColor(person_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            person_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            person_img = person_img / 255.0
            person_img = torch.FloatTensor(person_img).permute(2, 0, 1).unsqueeze(0)
            person_img = person_img.to(self.device)
            
            with torch.no_grad():
                outputs = self.classification_model(person_img)
                probs = torch.softmax(outputs, dim=1)
                
                child_prob = float(probs[0, 1])
                therapist_prob = float(probs[0, 0])
                
                # Initial classification
                if child_prob > self.confidence_threshold:
                    label, confidence = 'Child', child_prob
                elif therapist_prob > self.confidence_threshold:
                    label, confidence = 'Therapist', therapist_prob
                else:
                    label = 'Child' if child_prob > therapist_prob else 'Therapist'
                    confidence = max(child_prob, therapist_prob)
                
                # Apply temporal smoothing
                return self._smooth_classification(track_id, label, confidence)
                    
        except Exception as e:
            self.logger.error(f"Error in classification: {e}")
            return None, 0

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_display: bool = True):
        """Process video with tracking and classification"""
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

                # Get detections
                detections = self.detect_persons(frame)
                tracks = []  # Initialize empty tracks list
                
                # Update tracks
                if len(detections) > 0:
                    # Convert detections to format expected by DeepSort
                    detection_list = []
                    for det in detections:
                        detection_list.append(([det[0], det[1], det[2] - det[0], det[3] - det[1]], det[4], 'person'))
                    
                    # Get updated tracks
                    tracks = self.tracker.update_tracks(detection_list, frame=frame)

                    # Process each track
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        ltwh = track.to_ltwh()
                        bbox = np.array([
                            ltwh[0], ltwh[1],
                            ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]
                        ])
                        
                        # Check for overlap with other tracks
                        overlap_detected = False
                        for other_track in tracks:
                            if other_track.track_id != track_id:
                                other_bbox = other_track.to_ltwh()
                                iou = self._calculate_iou(
                                    bbox, 
                                    np.array([other_bbox[0], other_bbox[1], 
                                            other_bbox[0] + other_bbox[2], 
                                            other_bbox[1] + other_bbox[3]])
                                )
                                if iou > 0.3:  # Significant overlap threshold
                                    overlap_detected = True
                                    break
                        
                        if not overlap_detected:
                            # Only classify if no significant overlap
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
        tracker = EnhancedPersonTracker(
            model_path='best_model.pth',
            confidence_threshold=0.6
        )
        return tracker
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_uploaded_video(tracker, video_file):
    """Process uploaded video file"""
    # Save uploaded file to temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    # Process video
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error opening video file")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video writer
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
    
    # Progress bar
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames
    processed_frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get detections and process frame
        detections = tracker.detect_persons(frame)
        if len(detections) > 0:
            detection_list = []
            for det in detections:
                detection_list.append(([det[0], det[1], det[2] - det[0], det[3] - det[1]], det[4], 'person'))
            
            tracks = tracker.tracker.update_tracks(detection_list, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltwh = track.to_ltwh()
                bbox = np.array([
                    ltwh[0], ltwh[1],
                    ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]
                ])
                
                label, confidence = tracker.predict_person(frame, bbox, track_id)
                if label is not None:
                    tracker.draw_detection(frame, bbox, track_id, label, confidence)
        
        out.write(frame)
        progress_bar.progress((i + 1) / frame_count)
    
    cap.release()
    out.release()
    
    return output_file.name

class VideoProcessor(VideoProcessorBase):
    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self._frame_lock = threading.Lock()
        self.frame_queue = FRAME_QUEUE
        self.result_queue = RESULT_QUEUE

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Process frame
            with self._frame_lock:
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
            logger.error(f"Error processing frame: {str(e)}")
            # Return original frame if processing fails
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def initialize_webrtc():
    """Initialize WebRTC with proper configuration and error handling"""
    try:
        rtc_configuration = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:numb.viagenie.ca"],
                    "username": "webrtc@live.com",
                    "credential": "muazkh",
                }
            ]}
        )
        
        return rtc_configuration
    except Exception as e:
        logger.error(f"Error initializing WebRTC: {str(e)}")
        return None

def setup_webcam_page(tracker):
    """Setup webcam page with proper error handling"""
    st.write("Webcam Feed")
    
    # Add status indicator
    status_placeholder = st.empty()
    status_placeholder.info("Initializing webcam...")
    
    try:
        rtc_config = initialize_webrtc()
        if rtc_config is None:
            status_placeholder.error("Failed to initialize WebRTC configuration")
            return
        
        # Add webcam options
        st.sidebar.subheader("Webcam Settings")
        video_quality = st.sidebar.selectbox(
            "Video Quality",
            ["Low", "Medium", "High"],
            index=1
        )
        
        # Convert quality settings to resolution
        quality_settings = {
            "Low": {"width": 640, "height": 480},
            "Medium": {"width": 854, "height": 480},
            "High": {"width": 1280, "height": 720}
        }
        
        webrtc_ctx = webrtc_streamer(
            key="person-tracking",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=lambda: VideoProcessor(tracker),
            media_stream_constraints={
                "video": {
                    "width": quality_settings[video_quality]["width"],
                    "height": quality_settings[video_quality]["height"],
                    "frameRate": {"ideal": 30}
                },
            },
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            status_placeholder.success("Webcam is active")
        else:
            status_placeholder.warning("Webcam is not active")
            
    except Exception as e:
        logger.error(f"Error in webcam setup: {str(e)}")
        status_placeholder.error(f"Error setting up webcam: {str(e)}")

def main():
    st.title("Person Tracking and Classification App")
    
    # Sidebar for app options
    st.sidebar.title("Settings")
    input_option = st.sidebar.radio(
        "Choose Input Source",
        ["Upload Video", "Use Webcam"]
    )
    
    # Load model with error handling
    try:
        with st.spinner("Loading model..."):
            tracker = EnhancedPersonTracker(
                model_path='best_model.pth',
                confidence_threshold=0.6
            )
            st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    if input_option == "Upload Video":
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Add a process button
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    try:
                        output_path = process_uploaded_video(tracker, uploaded_file)
                        st.success("Video processed successfully!")
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, 'rb') as file:
                            st.download_button(
                                label="Download processed video",
                                data=file,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
    
    else:  # Webcam option
        setup_webcam_page(tracker)

if __name__ == "__main__":
    main()

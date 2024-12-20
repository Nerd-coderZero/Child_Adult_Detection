
import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
from typing import Tuple, List, Dict, Optional

class EnhancedPersonTracker:
    def __init__(
            self,
            model_path: str = 'best_model.pth',
            confidence_threshold: float = 0.70,
            input_size: tuple = (416, 416),  # Added parameter
            close_up_min_pixels: int = 60,
            close_up_max_pixels: int = 250,
            close_up_aspect_ratio_bounds: Tuple[float, float] = (0.4, 1.8),
            boundary_margin_ratio: float = 0.15
        ):
            # Setup logging and device
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Store input size for use in other methods
            self.input_size = input_size

            # Core parameters - adjusted for better detection
            self.detection_params = {
                'max_size_ratio': 0.80,  # Increased to handle close objects
                'min_pixels': 80,  # Increased minimum size to reduce false positives
                'aspect_ratio_bounds': (0.3, 2.0),  # Tightened aspect ratio bounds
                'padding_ratio': 0.05,  # Reduced padding to prevent overlap
                'temporal_smoothing': 7  # Frames for temporal smoothing
            }

            # Additional parameters for close-up handling
            self.close_up_min_pixels = close_up_min_pixels
            self.close_up_max_pixels = close_up_max_pixels
            self.close_up_aspect_ratio_bounds = close_up_aspect_ratio_bounds
            self.confidence_threshold = confidence_threshold
            self.boundary_margin_ratio = boundary_margin_ratio

            # Initialize classification history
            self.classification_history = {}

            # Initialize models and tracker
            try:
                self.classification_model = self._load_model(model_path)
                self.detector = self._load_detector()  # This will use input_size
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

def main():
    """Main function with enhanced user interface, webcam support, and error handling"""
    print("\n=== Therapy Session Analysis System ===")
    print("Initializing tracking system...")

    try:
        # Initialize tracker with configurable parameters
        tracker = EnhancedPersonTracker(
            model_path='best_model.pth',
            confidence_threshold=0.6,
            input_size=(416, 416)
        )

        # Process videos in the directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path('.').glob(f'*{ext}'))

        # Display options including webcam
        print("\nAvailable options:")
        print("0. Webcam")
        if video_files:
            print("\nRecorded therapy session videos:")
            for i, video in enumerate(video_files):
                print(f"{i+1}. {video.name}")
        else:
            print("\nNo video files found in the current directory!")

        # Enhanced user input handling
        while True:
            try:
                user_input = input("\nSelect option (0 for webcam, number for video, or 'q' to quit): ")
                if user_input.lower() == 'q':
                    print("Exiting program.")
                    return

                selection = int(user_input)
                if selection == 0:  # Webcam option
                    video_path = 0  # OpenCV uses 0 for default webcam
                    output_path = "webcam_output.mp4"
                    break
                elif 0 < selection <= len(video_files):
                    video_path = str(video_files[selection - 1])
                    output_path = f"analyzed_{Path(video_path).stem}{Path(video_path).suffix}"
                    break
                print(f"Invalid selection. Please enter a number between 0 and {len(video_files)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")

        print("\n=== Processing Configuration ===")
        if video_path == 0:
            print("Input: Webcam")
        else:
            print(f"Input video: {video_path}")
        print(f"Output will be saved to: {output_path}")
        print("\nControls:")
        print("- Press 'q' to quit processing")
        print("- Press 'p' to pause/resume")
        print("- Press 's' to save a screenshot")
        
        if video_path == 0:
            print("\nInitializing webcam...")
            # Test webcam availability
            test_cap = cv2.VideoCapture(0)
            if not test_cap.isOpened():
                test_cap.release()
                raise Exception("Could not access webcam. Please check if it's connected and not in use by another application.")
            test_cap.release()
            print("Webcam initialized successfully!")
        
        print("\nProcessing video...")

        # Process the video with enhanced error handling
        tracker.process_video(video_path, output_path, show_display=True)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. If using webcam, check if it's properly connected and not in use")
        print("2. For video files, check if the file is corrupted")
        print("3. Verify sufficient system memory")
        print("4. Ensure model file 'best_model.pth' exists")
        print("5. Check GPU/CUDA compatibility if using GPU")
    finally:
        print("\nProcessing complete. Check output directory for results.")

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import numpy as np
from sort import Sort
import torch
from inference_sdk import InferenceHTTPClient
import tempfile
from PIL import Image
import io

class PersonTracker:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize YOLOv5 model
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.to(self.device)
        self.yolo_model.conf = confidence_threshold
        
        # Initialize SORT tracker
        self.tracker = Sort(
            max_age=30,
            min_hits=5,
            iou_threshold=0.25
        )
        
        # Initialize Roboflow client
        self.classifier_client = InferenceHTTPClient(
            api_url="https://classify.roboflow.com",
            api_key="ElqVQt7rQnBAEBOdYRjv"
        )
        
        # Track classifications
        self.person_classifications = {}
        self.classification_history = {}
        self.history_size = 5
    
    def detect_person(self, frame):
        """Detect persons using YOLOv5."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(frame_rgb)
        
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) == 0:  # person class
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    
    def classify_person(self, frame, bbox):
        """Classify person using Roboflow model."""
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Extract person image
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None, 0.0
            
            # Convert to PIL Image and save to bytes
            pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Get prediction from Roboflow
            result = self.classifier_client.infer(
                img_byte_arr,
                model_id="child-adult-classifier/1"
            )
            
            # Parse result
            prediction = result['predicted_class']
            confidence = result['confidence']
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return None, 0.0
    
    def update_tracking(self, frame, detections):
        """Update tracking with classifications."""
        if len(detections) == 0:
            return []
        
        tracked_objects = self.tracker.update(detections)
        results = []
        
        for track in tracked_objects:
            track_id = int(track[4])
            bbox = track[:4]
            
            # Get classification
            classification, confidence = self.classify_person(frame, bbox)
            
            if classification is not None:
                if track_id not in self.person_classifications:
                    self.person_classifications[track_id] = {
                        'class': classification,
                        'confidence': confidence
                    }
                elif confidence > self.person_classifications[track_id]['confidence']:
                    self.person_classifications[track_id] = {
                        'class': classification,
                        'confidence': confidence
                    }
            
            current_class = (self.person_classifications.get(track_id, {})
                           .get('class', 'adult'))  # Default to adult
            
            results.append({
                'track_id': track_id,
                'bbox': bbox,
                'class': current_class,
                'confidence': self.person_classifications.get(
                    track_id, {}).get('confidence', 0.0)
            })
        
        return results
    
    def draw_results(self, frame, results):
        """Draw tracking results on frame."""
        for result in results:
            bbox = result['bbox']
            track_id = result['track_id']
            classification = result['class']
            confidence = result.get('confidence', 0.0)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on classification
            color = (0, 255, 0) if classification == 'adult' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{classification} ({confidence:.2f}) ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw statistics
        adult_count = sum(1 for r in results if r['class'] == 'adult')
        child_count = sum(1 for r in results if r['class'] == 'child')
        cv2.putText(frame, f"Adults: {adult_count} Children: {child_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

def main():
    st.title("Person Tracker with Age Classification")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Initialize tracker
    tracker = PersonTracker(confidence_threshold=confidence_threshold)
    
    # File uploader
    video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        # Video processing
        cap = cv2.VideoCapture(tfile.name)
        
        # Create a placeholder for the video
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and track persons
            detections = tracker.detect_person(frame)
            results = tracker.update_tracking(frame, detections)
            
            # Draw results
            frame = tracker.draw_results(frame, results)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            stframe.image(frame)
            
        cap.release()
    
    else:
        st.write("Please upload a video file")

if __name__ == "__main__":
    main()

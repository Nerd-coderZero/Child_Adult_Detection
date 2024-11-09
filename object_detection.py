import cv2
import numpy as np
import torch
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient
from sort import Sort

class PersonTracker:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize YOLOv5 model
        print("Loading YOLOv5 model...")
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo_model.to(self.device)
        self.yolo_model.conf = confidence_threshold
        
        # Initialize SORT tracker
        print("Initializing SORT tracker...")
        self.tracker = Sort(
            max_age=30,
            min_hits=5,
            iou_threshold=0.25
        )
        
        # Initialize Roboflow client
        print("Initializing Roboflow client...")
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
    
    def update_classification_history(self, track_id, classification, confidence):
        """Update classification history with temporal smoothing."""
        if track_id not in self.classification_history:
            self.classification_history[track_id] = []
        
        # Add new classification to history
        self.classification_history[track_id].append((classification, confidence))
        
        # Keep only last N classifications
        if len(self.classification_history[track_id]) > self.history_size:
            self.classification_history[track_id].pop(0)
        
        # Exponential weighted voting
        votes = {'adult': 0.0, 'child': 0.0}
        weights = [0.6 ** i for i in range(len(self.classification_history[track_id]))]
        
        for (cls, conf), weight in zip(self.classification_history[track_id], weights):
            votes[cls] += conf * weight
        
        # Normalize votes
        total_weight = sum(weights)
        for cls in votes:
            votes[cls] /= total_weight
        
        final_class = max(votes.items(), key=lambda x: x[1])[0]
        avg_confidence = votes[final_class]
        
        return final_class, avg_confidence

    def classify_person(self, frame, bbox):
        """Enhanced classification with multiple samples."""
        try:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Add margin to bounding box
            margin = int((y2 - y1) * 0.15)  # 15% margin
            y1 = max(0, y1 - margin)
            y2 = min(frame.shape[0], y2 + margin)
            x1 = max(0, x1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 30 or (y2 - y1) < 30:
                return None, 0.0
            
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None, 0.0
            
            # Create two versions of the image (original and flipped)
            pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            flipped_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Get predictions for both versions
            results = []
            for img in [pil_img, flipped_img]:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                result = self.classifier_client.infer(
                    img_byte_arr,
                    model_id="child-adult-classifier/1"
                )
                results.append((result['predicted_class'], result['confidence']))
            
            # Combine predictions
            if results[0][0] == results[1][0]:  # If both predictions agree
                return results[0][0], max(results[0][1], results[1][1])
            else:  # If predictions disagree, return the one with higher confidence
                return max(results, key=lambda x: x[1])
                
        except Exception as e:
            print(f"Error in classification: {e}")
            return None, 0.0

    def update_tracking(self, frame, detections):
        """Update tracking with improved classification logic."""
        if len(detections) == 0:
            return []
        
        tracked_objects = self.tracker.update(detections)
        results = []
        
        for track in tracked_objects:
            track_id = int(track[4])
            bbox = track[:4]
            
            # Get classification and confidence
            classification, confidence = self.classify_person(frame, bbox)
            
            if classification is not None:
                # Update classification history and get final classification
                final_class, avg_confidence = self.update_classification_history(
                    track_id, classification, confidence
                )
                
                if track_id not in self.person_classifications:
                    self.person_classifications[track_id] = {
                        'class': final_class,
                        'confidence': avg_confidence
                    }
                elif avg_confidence > self.person_classifications[track_id]['confidence']:
                    self.person_classifications[track_id] = {
                        'class': final_class,
                        'confidence': avg_confidence
                    }
            
            current_class = (self.person_classifications.get(track_id, {})
                            .get('class', 'adult'))
            
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
    # Initialize tracker
    tracker = PersonTracker(confidence_threshold=0.5)
    
    # Open video capture
    print("Opening video capture...")
    cap = cv2.VideoCapture('2.mp4')  # Use 0 for webcam or provide video file path
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect and track persons
        detections = tracker.detect_person(frame)
        results = tracker.update_tracking(frame, detections)
        
        # Draw results
        frame = tracker.draw_results(frame, results)
        
        # Display the frame
        cv2.imshow('Person Tracker', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Enhanced Person Tracker for Therapy Sessions
## Overview
This project implements an advanced person detection and tracking system specifically designed for therapy session analysis. Using YOLOv8 for detection, DeepSort for tracking, and a custom EfficientNet-based classification model, the system can identify and track therapists and children in video footage while maintaining consistent IDs throughout the session.

## Key Features
- **Enhanced Person Detection**: Utilizes YOLOv8 with optimized parameters for robust person detection
- **Reliable Tracking**: Implements DeepSort with customized parameters for consistent tracking in therapy settings
- **Advanced Classification**: Uses EfficientNet-B0 for accurate therapist/child classification
- **Temporal Smoothing**: Implements classification smoothing to reduce flickering and improve stability
- **Overlap Prevention**: Handles overlapping detections to prevent misclassification
- **Adaptive Processing**: Supports both recorded videos and real-time webcam input
- **Enhanced Visualization**: Clear bounding boxes and labels with confidence scores

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- Deep Sort Real-Time
- Additional dependencies listed in `requirements.txt`

## Installation
1. Clone the repository
```bash
git clone https://github.com/Nerd-coderZero/Child_Adult_Detection
cd Child_Adult_Detection
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download required models:
   - Place the classification model (`best_model.pth`) in the project root directory
   - YOLOv8 model will be downloaded automatically on first run

## Usage
Run the main script:
```bash
python object_detection.py
```

The program will present options for:
- Live webcam processing (Option 0)
- Processing recorded video files (Options 1+)

### Controls During Processing
- `q`: Quit processing
- `p`: Pause/resume
- `s`: Save screenshot

## Configuration
The `EnhancedPersonTracker` class accepts several parameters for customization:
```python
tracker = EnhancedPersonTracker(
    model_path='best_model.pth',          # Path to classification model
    confidence_threshold=0.70,            # Minimum confidence for classification
    input_size=(416, 416),               # Input size for models
    close_up_min_pixels=60,              # Minimum size for close-up detection
    close_up_max_pixels=250,             # Maximum size for close-up detection
    boundary_margin_ratio=0.15           # Margin for boundary detection
)
```

## Output
- Processes videos with real-time visualization
- Saves analyzed videos with annotations
- Displays:
  - Bounding boxes (green for therapists, orange for children)
  - Labels with confidence scores
  - Tracking IDs
  - Classification results

## Technical Details
### Detection System
- Implements YOLOv8 with optimized parameters
- Includes preprocessing for improved detection quality
- Handles various lighting conditions and scenarios

### Tracking System
- Uses DeepSort with custom parameters for therapy settings
- Implements temporal smoothing for stable tracking
- Handles occlusions and reappearances effectively

### Classification System
- EfficientNet-B0 based architecture
- Binary classification (Therapist/Child)
- Implements confidence thresholding
- Includes temporal smoothing for stable results

## Error Handling
- Comprehensive error handling and logging
- Detailed error messages and troubleshooting steps
- Graceful fallbacks for various failure scenarios

## Troubleshooting
If you encounter issues:
1. Check webcam connectivity and permissions
2. Verify video file integrity
3. Ensure sufficient system memory
4. Confirm model file existence
5. Check GPU/CUDA compatibility if using GPU

## Limitations
- Requires good lighting conditions for optimal performance
- May need adjustment of parameters for different environments
- Classification accuracy may vary with unusual poses or positions

## Future Improvements
- Integration with additional tracking algorithms
- Support for multi-camera setups
- Enhanced classification model training
- Real-time analytics and reporting
- Integration with therapy session management systems

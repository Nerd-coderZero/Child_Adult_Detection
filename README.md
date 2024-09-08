# Person Detection and Tracking with Unique ID Assignment

## Overview

This project involves the detection and tracking of individuals (adults and children) in a video. Using TensorFlow Object Detection and a custom classification approach, the script assigns unique IDs to detected persons and tracks their movements across video frames. The project has been designed to assist in environments like therapy sessions where tracking children with Autism Spectrum Disorder is critical.

## Features
Person Detection: Detects persons in each frame of the video using a pre-trained TensorFlow Object Detection model.
Child/Adult Classification: Distinguishes between children and adults using a height-based threshold from pose landmarks (fallbacks to a TensorFlow classification model if needed).
Person Tracking: Assigns unique IDs to each detected person and tracks their movement across video frames.
Re-identification: Reassigns the correct ID to persons after occlusions or reappearance in the frame.
Output: Produces a video file with bounding boxes and labels (Child/Adult) along with unique IDs for each person.

#### Installation

1. Clone the repository:

git clone <https://github.com/Nerd-coderZero/Child_Adult_Detection>  (may not yet made git repository so just copy given zip folder to your project folder)
cd <Child_Adult_Detection>

2. Install dependencies:(Make sure you are into Virtual Environment of project folder to not mix dependencies)
Ensure you have Python installed, and install the dependencies listed in requirements.txt:

pip install -r requirements.txt

3. Download and set up models:

Object Detection Model: Place the TensorFlow object detection model in the appropriate directory (ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model).
Child/Adult Classification Model: Ensure the child_adult_model.h5 file is available in the project root directory.
Usage
Run the script with the sample video:
    
python object_detection.py 


Output

    The output video will display bounding boxes around detected individuals with their assigned IDs and labels (Child/Adult). The output video will be saved as output.mp4.

Notes
    
The classification relies on pose estimation and height-based heuristics. In cases of seated children and adults, there are still some classification inaccuracies.
Custom models or further tuning can improve classification accuracy.


```python

```

import streamlit as st
from object_detection import EnhancedPersonTracker
from io import BytesIO

def main():
    st.title("Therapist and Child Detection and Tracking")

    # Load the video file
    video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
        # Initialize the person tracker
        tracker = EnhancedPersonTracker(model_path='best_model.pth')

        # Process the video and get the output bytes
        output_bytes = tracker.process_video(video_file, show_display=False)

        # Display the video using Streamlit's built-in video player
        st.video(output_bytes)

if __name__ == "__main__":
    main()

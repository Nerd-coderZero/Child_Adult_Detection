import streamlit as st
from object_detection import EnhancedPersonTracker

def main():
    st.title("Therapist and Child Detection and Tracking")

    # Add an option to choose between video file upload and webcam
    tracking_mode = st.radio("Choose Tracking Mode", ("Upload Video", "Use Webcam"))

    if tracking_mode == "Upload Video":
        # Video file upload logic
        video_file = st.file_uploader("Upload a video file", type=["mp4"])
        if video_file is not None:
            # Process the uploaded video
            tracker = EnhancedPersonTracker(model_path='best_model.pth')
            output_bytes = tracker.process_video(video_file, show_display=False)
            st.video(output_bytes)
    elif tracking_mode == "Use Webcam":
        # Webcam logic
        st.write("Using webcam for real-time tracking...")
        # Implement real-time tracking using the webcam
        # (this may require additional resource handling and error checking)

if __name__ == "__main__":
    main()

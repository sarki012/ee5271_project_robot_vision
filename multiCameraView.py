# Erik Sarkinen
# Multi-Camera Viewer

import cv2
import numpy as np

def main():
    # Define camera indices for the 3 cameras
    # NOTE: This script must be run ON the Raspberry Pi to see the Pi's cameras.
    # If run on a PC, it will access the PC's webcams.
    
    # Scan for available cameras (indices 0-9)
    camera_indices = []
    print("Scanning for cameras...")
    for i in range(10):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if temp_cap.isOpened():
            # Set MJPG to ensure we can read a frame without bandwidth issues
            temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            ret, _ = temp_cap.read()
            if ret:
                print(f"Found camera at index {i}")
                camera_indices.append(i)
            temp_cap.release()
        if len(camera_indices) >= 3:
            break
            
    caps = []

    # Open video capture for each camera
    for index in camera_indices:
        # Force V4L2 backend to avoid GStreamer errors with USB cameras on Pi
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # Set MJPG to save USB bandwidth
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # Set a reasonable resolution for display to fit all on screen
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if cap.isOpened():
            print(f"Camera {index} opened successfully.")
        else:
            print(f"Warning: Camera {index} failed to open. Is another process (like robotMain) using it?")
        caps.append(cap)

    print("Press 'q' to quit.")

    while True:
        frames = []
        
        for i, cap in enumerate(caps):
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Resize frame to ensure they match for stacking
                    # (320, 240) matches the set resolution
                    frame = cv2.resize(frame, (320, 240))
                    # Add label
                    cv2.putText(frame, f"Cam {camera_indices[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    frames.append(frame)
                else:
                    # Frame read failed
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(blank, f"No Signal {i}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 0, 255), 2)
                    frames.append(blank)
            else:
                # Camera not opened
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, f"Cam {i} Error", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2)
                frames.append(blank)

        # Concatenate frames horizontally
        if frames:
            combined_image = np.hstack(frames)
            cv2.imshow("Multi-Camera View", combined_image)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all captures
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
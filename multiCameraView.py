# Erik Sarkinen
# Multi-Camera Viewer

import cv2
import numpy as np
import sys
import math
from collections import deque

def task4():
    # Define camera indices for the 3 cameras
    # NOTE: This script must be run ON the Raspberry Pi to see the Pi's cameras.
    # If run on a PC, it will access the PC's webcams.
    
    # Scan for available cameras (indices 0-9)
    camera_indices = []
    print("Scanning for cameras...")
    for i in range(10):
        # Use DirectShow on Windows for better webcam compatibility
        backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_DSHOW
        temp_cap = cv2.VideoCapture(i, backend)
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
            
    if not camera_indices:
        print("Error: No cameras detected. Check USB connections, Windows Camera Privacy settings, or if another app is using them.")
        sys.exit(1)
            
    # Define labels for specific camera indices to differentiate them
    camera_labels = {
        1: "Left",
        3: "Right"
    }

    # Find the list indices corresponding to the left and right cameras (or first two available)
    left_cam_idx = -1
    right_cam_idx = -1
    
    # Map hardware indices to list indices
    hw_idx_to_list_idx = {hw_idx: i for i, hw_idx in enumerate(camera_indices)}
    
    if 1 in hw_idx_to_list_idx and 3 in hw_idx_to_list_idx:
        left_cam_idx = hw_idx_to_list_idx[1]
        right_cam_idx = hw_idx_to_list_idx[3]
    elif len(camera_indices) >= 2:
        # Fallback to first two cameras found if 1 and 3 are not available
        left_cam_idx = 0
        right_cam_idx = 1
        print(f"Cameras 1 and 3 not found together. Using indices {camera_indices[0]} and {camera_indices[1]} for stereo matching.")

    caps = []

    # Open video capture for each camera
    for index in camera_indices:
        # Force V4L2 backend on Linux to avoid GStreamer errors with USB cameras on Pi.
        # Use DirectShow on Windows for better webcam compatibility.
        backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(index, backend)
        # Set MJPG to save USB bandwidth
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # Set a reasonable resolution for display to fit all on screen
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if cap.isOpened():
            print(f"Camera {index} opened successfully.")
        else:
            print(f"Warning: Camera {index} failed to open. Is another process (like robotMain) using it?")
        caps.append(cap)

    print("Press 'q' to quit.")
    # Create a window and move it to the top-left corner (0,0)
    cv2.namedWindow("Multi-Camera View", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Multi-Camera View", 0, 0)

    # --- Stereo Rectification Setup ---
    # NOTE: Update the remaining fx, fy, Cy, R, and T values with the output from calib.py
    rectify_size = (480, 360)
    
    # Intrinsic matrices dynamically scaled for the 480x360 resolution.
    # Assuming original calibration was at 1280x720 yielding fx=1200:
    # 1200 * (480 / 1280) = 450 pixels.
    K_left = np.array([[450.0, 0.0, 240.0],
                       [0.0, 450.0, 180.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    D_left = np.zeros(5, dtype=np.float64)
    
    K_right = np.array([[450.0, 0.0, 240.0],
                        [0.0, 450.0, 180.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    D_right = np.zeros(5, dtype=np.float64)
    
    # Extrinsic parameters (Rotation and Translation between the two cameras)
    # Cameras are parallel (R = Identity). Camera1 is 14.6 cm to the right of Camera0.
    R_ext = np.eye(3, dtype=np.float64)
    T_ext = np.array([[14.6], [0.0], [0.0]], dtype=np.float64) 
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, rectify_size, R_ext, T_ext
    )
    
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, rectify_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, rectify_size, cv2.CV_16SC2)
    # ----------------------------------

    misc_deque = deque(maxlen=20)
    while True:
        frames = []
        gray_frames = {}
        
        for i, cap in enumerate(caps):
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Resize frame to ensure they match for stacking
                    # (480, 360) matches the set resolution (1.5x of 320x240)
                    frame = cv2.resize(frame, (480, 360))

                    # Apply stereo rectification mapping
                    if i == left_cam_idx:
                        frame = cv2.remap(frame, left_map1, left_map2, cv2.INTER_LINEAR)
                    elif i == right_cam_idx:
                        frame = cv2.remap(frame, right_map1, right_map2, cv2.INTER_LINEAR)

                    # Prepare gray image
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # If left or right camera, save clean gray copy for matching
                    if i == left_cam_idx or i == right_cam_idx:
                        gray_frames[i] = gray.copy()

                    # Add label
                    idx = camera_indices[i]
                    label = camera_labels.get(idx, f"Cam {idx}")
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    frames.append(frame)
                else:
                    # Frame read failed
                    blank = np.zeros((360, 480, 3), dtype=np.uint8)
                    cv2.putText(blank, f"No Signal {i}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 0, 255), 2)
                    frames.append(blank)
            else:
                # Camera not opened
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(blank, f"Cam {i} Error", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2)
                frames.append(blank)

        # Concatenate frames horizontally
        if frames:
            combined_image = np.hstack(frames)
              # --- Feature Matching Visualization ---
            match_vis_img = None
            if left_cam_idx != -1 and right_cam_idx != -1 and left_cam_idx in gray_frames and right_cam_idx in gray_frames:
                img_left_gray = gray_frames[left_cam_idx]
                img_right_gray = gray_frames[right_cam_idx]
                
                # Find matches on clean grayscale images before drawing on them
                sx1, sx2, _, _, _ = find_match(img_left_gray, img_right_gray, show_window=False)
                x1 = sx1
                x2 = sx2

                if len(x1) > 0:
                    disparities = np.abs(x1[:, 0] - x2[:, 0])
                    avg_disparity = np.mean(disparities)
                    print(f"Average Disparity: {avg_disparity:.2f} pixels")
                    
                    if avg_disparity > 0:
                        # Z = (scaled_focal_length * baseline_meters) / disparity
                        Z = (450.0 * 0.146) / avg_disparity
                        print(f"Z = {Z:.6f}")

                # --- Prepare visualization images with contours ---
                # Convert to BGR to draw color contours
                vis_img_left = cv2.cvtColor(img_left_gray, cv2.COLOR_GRAY2BGR)
                vis_img_right = cv2.cvtColor(img_right_gray, cv2.COLOR_GRAY2BGR)

                try:
                    # Use RANSAC to find inliers and visualize them on the images with contours
                    h, w = vis_img_left.shape[:2]
                    combined_vis = np.hstack((vis_img_left, vis_img_right))

                    inliers_mask = None
                    if len(x1) >= 4:
                        _, mask = cv2.findHomography(x1, x2, cv2.RANSAC, 5.0)
                        if mask is not None:
                            inliers_mask = mask.ravel()

                    for i in range(len(x1)):
                        if inliers_mask is not None and inliers_mask[i]:
                            pt1 = (int(x1[i][0]), int(x1[i][1]))
                            pt2 = (int(x2[i][0] + w), int(x2[i][1]))
                            color = (0, 255, 0)
                            misc_deque.append((pt1, pt2, color))

                    for (pt1, pt2, color) in misc_deque:
                        cv2.line(combined_vis, pt1, pt2, color, 2)
                    match_vis_img = combined_vis
                except Exception as e:
                    # Handle cases where matching fails
                    h, w = img_left_gray.shape
                    match_vis_img = np.zeros((h, w * 2, 3), dtype=np.uint8)

            # Stack the match visualization below the live streams
            if match_vis_img is not None:
                # Resize match visualization to match the width of the live streams
                target_w = combined_image.shape[1]
                h, w = match_vis_img.shape[:2]
                scale = target_w / w
                resized_match_vis = cv2.resize(match_vis_img, (target_w, int(h * scale)))
                if len(resized_match_vis.shape) == 2:
                    resized_match_vis = cv2.cvtColor(resized_match_vis, cv2.COLOR_GRAY2BGR)
                combined_image = np.vstack((combined_image, resized_match_vis))

            cv2.imshow("Multi-Camera View", combined_image)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all captures
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    return combined_image

def find_match(img1, img2, show_window=True):
    # To do
    # Create a SIFT (Scale-Invariant Feature Transform) object
    sift = cv2.SIFT_create()   
    '''
    sift.detectAndCompute: locates keypoints and calculates feature descriptors, which are
    128-dimensional vectors.
    '''
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        return np.array([]), np.array([]), keypoints1, keypoints2, []

    '''
    find the 2 closest matches in img2 for every feature in img1.
    '''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    '''
    The following code compares the distance of the best match (distance[i][0]) to the 
    second-best match (distance[i][1]). If the best match is much closer 
    (less than 0.75*distance) than the second-best, it is a match.
    '''
    x1 = []
    x2 = []
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                x1.append(keypoints1[m.queryIdx].pt)
                x2.append(keypoints2[m.trainIdx].pt)
                good_matches.append(m)

    if show_window:
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
    x_array1 = np.array(x1)
    x_array2 = np.array(x2)
    return x_array1, x_array2, keypoints1, keypoints2, good_matches

if __name__ == "__main__":
    task4()
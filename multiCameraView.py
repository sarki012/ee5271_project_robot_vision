# Erik Sarkinen
# Multi-Camera Viewer

import cv2
import numpy as np
import numpy as np
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
        # Force V4L2 backend to avoid GStreamer errors with USB cameras on Pi
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
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

    circle_deque = deque(maxlen=2)
    misc_deque = deque(maxlen=100)
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
                cx1, cx2 = find_circle_match(img_left_gray, img_right_gray)
                
                x1_list = []
                x2_list = []
                if len(sx1) > 0:
                    x1_list.append(sx1)
                    x2_list.append(sx2)
                if len(cx1) > 0:
                    x1_list.append(cx1)
                    x2_list.append(cx2)
                
                if x1_list:
                    x1 = np.vstack(x1_list)
                    x2 = np.vstack(x2_list)
                else:
                    x1 = np.array([])
                    x2 = np.array([])

                # --- Prepare visualization images with contours ---
                # Convert to BGR to draw color contours
                vis_img_left = cv2.cvtColor(img_left_gray, cv2.COLOR_GRAY2BGR)
                vis_img_right = cv2.cvtColor(img_right_gray, cv2.COLOR_GRAY2BGR)

                # Find and draw contours on left image
                blurred_left = cv2.blur(img_left_gray, (3, 3))
                canny_left = cv2.Canny(blurred_left, 50, 150)
                contours_left, _ = cv2.findContours(canny_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img_left, contours_left, -1, (0, 0, 255), 2)

                # Find and draw contours on right image
                blurred_right = cv2.blur(img_right_gray, (3, 3))
                canny_right = cv2.Canny(blurred_right, 50, 150)
                contours_right, _ = cv2.findContours(canny_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img_right, contours_right, -1, (0, 0, 255), 2)

                try:
                    # Use RANSAC to find inliers and visualize them on the images with contours
                    _, _, best_inliers = align_image_using_feature(x1, x2, 5.0, 200, vis_img_left, vis_img_right)
                    
                    h, w = vis_img_left.shape[:2]
                    combined_vis = np.hstack((vis_img_left, vis_img_right))

                    if best_inliers is not None:
                        for i in range(len(x1)):
                            pt1 = (int(x1[i][0]), int(x1[i][1]))
                            pt2 = (int(x2[i][0] + w), int(x2[i][1]))
                            color = (0, 255, 0) if best_inliers[i] else (255, 0, 0)
                            
                            is_circle = False
                            if len(cx1) > 0:
                                dists = np.linalg.norm(cx1 - x1[i], axis=1)
                                if np.min(dists) < 1.0:
                                    is_circle = True
                            
                            if is_circle:
                                circle_deque.append((pt1, pt2, color, is_circle))
                            else:
                                misc_deque.append((pt1, pt2, color, is_circle))

                    all_matches = list(circle_deque) + list(misc_deque)
                    for (pt1, pt2, color, is_circle) in all_matches:
                        cv2.line(combined_vis, pt1, pt2, color, 2)
                        if is_circle:
                            cv2.circle(combined_vis, pt1, 15, (0, 255, 255), 2)
                            cv2.circle(combined_vis, pt2, 15, (0, 255, 255), 2)
                    
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

def find_circle_match(img1, img2):
    circles1 = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=300)
    circles2 = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=300)
    
    x1 = []
    x2 = []
    
    if circles1 is not None and circles2 is not None:
        circles1 = np.round(circles1[0, :]).astype("int")
        circles2 = np.round(circles2[0, :]).astype("int")
        
        for (x1_c, y1_c, r1_c) in circles1:
            best_match = None
            min_dist = float('inf')
            for (x2_c, y2_c, r2_c) in circles2:
                dist = abs(y1_c - y2_c) + abs(r1_c - r2_c)
                if dist < 50: # Threshold for match
                    if dist < min_dist:
                        min_dist = dist
                        best_match = (x2_c, y2_c)
            if best_match:
                x1.append([x1_c, y1_c])
                x2.append([best_match[0], best_match[1]])
    return np.array(x1), np.array(x2)

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

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter, img1=None, img2=None):
    # To do
    '''
    RANSAC algorithm for Affine Transformation fitting. 
    '''
    inliers = []
    distance = []
    vis = None
    num_inliers = 0
    best_inliers = None
    max_inlier_count = 0
    num_samples = 3 
    '''
    The goal is to find the best 2D transformation (rotation, scale, translation, shear) 
    that aligns the points in x1 with the points in x2, even if many of the initial matches 
    are incorrect (outliers).
    '''
    if len(x1) < num_samples:
        if img1 is not None and img2 is not None and len(x1) > 0:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
            img1_c = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_c = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            vis[:h1, :w1] = img1_c
            vis[:h2, w1:w1+w2] = img2_c
            best_inliers = [True] * len(x1)
            for i in range(len(x1)):
                pt1 = (int(x1[i][0]), int(x1[i][1]))
                pt2 = (int(x2[i][0] + w1), int(x2[i][1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            return np.eye(3), vis, best_inliers
        return np.eye(3), None, None

    for i in range(ransac_iter):
        # Select random samples
        index = np.random.choice(len(x1), num_samples, replace=False)
        points1 = x1[index]
        points2 = x2[index]
        
        '''
        Fit Affine Model to 3 points
        We want AFFINE_TEMP such that points2 = AFFINE_TEMP * points1 (in homogeneous coordinates)
        X_ARRAY * AFFINE_TRANSPOSE = Y_ARRAY  =>  AFFINE_TRANSPOSE = inv(X_ARRAY) * Y_ARRAY
        It calculates a temporary 2x3 affine matrix AFFINE_TEMP that perfectly maps the 3 
        randomly chosen source points (pts1) to their corresponding target points (pts2). This 
        is done by solving the linear system Y_ARRAY = AFFINE_TEMP * X_ARRAY.
        '''
        X_ARRAY = np.hstack((points1, np.ones((3, 1))))     # Inserts a column of 1's (x, y, 1), homogenous coordinates
        Y_ARRAY = points2
        try:
            if np.linalg.matrix_rank(X_ARRAY) < 3:    # Makes sure that all 3 points aren't on the same line
                continue
            AFFINE_TRANSPOSE = np.linalg.solve(X_ARRAY, Y_ARRAY)
            AFFINE_TEMP = AFFINE_TRANSPOSE.T # 2x3 matrix
        except np.linalg.LinAlgError:
            continue
        
        '''
        Transform all x1 points using M
        Once a candidate transformation (AFFINE_TEMP) is calculated from the 3 random points, 
        the code tests how well it works for all the other points. X_all becomes an (N, 3) 
        matrix where every row is (x, y, 1).
        X_all: The source points in homogeneous coordinates Nx3.
        AFFINE_TEMP: The 2x3 affine matrix calculated from the random sample.
        x2_predicted is an Nx2 matrix containing the predicted coordinates in the second image.
        '''
        X_all = np.hstack((x1, np.ones((len(x1), 1))))
        x2_predicted = X_all @ AFFINE_TEMP.T
        
        '''
        x2: where the points are. x2_predicted: where the points should be.
        If diff is small, count as an inlier.
        '''
        diff = x2 - x2_predicted
        distance = np.linalg.norm(diff, axis=1)
        
        '''
        Points with an error smaller than the threshold are marked as inliers. These are the "good" 
        matches that agree with the current model.
        distance: This is a 1D NumPy array containing the error for each point. The error is the 
        pixel distance between where a point from the first image is predicted to be in the second 
        image (using the temporary model AFFINE_TEMP) and where it actually is. 
        inliers is a mask with 
        '''
        inliers = distance < ransac_thr     # inliers is a boolean mask, True if distance < ransac_thr
        num_inliers = np.sum(inliers)

        '''
        It keeps track of the model that has the highest number of inliers. This is assumed to be 
        the correct model. num_inliers and best_inliers are boolean masks with True at the index
        where distance is < ransac_thr.
        '''  
        if num_inliers > max_inlier_count:
            max_inlier_count = num_inliers
            best_inliers = inliers
    
    # Re-fit with all inliers
    if best_inliers is not None and max_inlier_count >= 3:
        if img1 is not None and img2 is not None:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Create a color canvas to draw on
            vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
            
            # Place images on the canvas, converting to color if they are grayscale
            img1_c = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_c = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            
            vis[:h1, :w1] = img1_c
            vis[:h2, w1:w1+w2] = img2_c

            for i in range(len(x1)):
                pt1 = (int(x1[i][0]), int(x1[i][1]))
                pt2 = (int(x2[i][0] + w1), int(x2[i][1]))
                color = (0, 255, 0) if best_inliers[i] else (0, 0, 255)
                cv2.line(vis, pt1, pt2, color, 1)
            
            # cv2.imshow("RANSAC Inliers", vis)
            # cv2.waitKey(0)
        '''
        It extracts the actual coordinates of all the "good" points (inlier_x1, inlier_x2) using 
        the saved mask best_inliers.
        '''
        inlier_x1 = x1[best_inliers]
        inlier_x2 = x2[best_inliers]
        
        X_ARRAY = np.hstack((inlier_x1, np.ones((len(inlier_x1), 1))))
        Y_ARRAY = inlier_x2
        
        '''
        It takes all the inliers found (which could be hundreds of points) and performs a 
        Least Squares fit. Unlike solve (which hits 3 points exactly), lstsq finds the 
        transformation that minimizes the average error across all valid points. This produces a 
        much more accurate and stable matrix.
        '''
        res = np.linalg.lstsq(X_ARRAY, Y_ARRAY, rcond=None)
        AFFINE_TRANSPOSE = res[0]
        A_AFFINE = AFFINE_TRANSPOSE.T
        '''
        It formats the result into a standard 3x3 homogeneous affine matrix 
        (adding [0, 0, 1] at the bottom) so it can be used for image warping later.
        '''
        A = np.vstack((A_AFFINE, [0, 0, 1]))
    else:
        print("RANSAC failed.")
        A = np.eye(3)       # Return identity matrix.
        
    print(f"Number of inliers found: {max_inlier_count}")
    print("Affine Transformation Matrix A:\n", A)
    return A, vis, best_inliers
def visualize_find_match(img1, img2, x1, x2, img_h=500, mask=None):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    
    if len(img1_resized.shape) == 2:
        img1_resized = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
    if len(img2_resized.shape) == 2:
        img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)

    img = np.hstack((img1_resized, img2_resized))
    cv2.putText(img, "Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Image 2", (img1_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    indices = range(x1.shape[0])
    if mask is not None:
        indices = np.argsort(mask)

    for i in indices:
        color = (255, 0, 0)
        if mask is not None and mask[i]:
            color = (0, 255, 255)
        pt1 = (int(x1[i, 0]), int(x1[i, 1]))
        pt2 = (int(x2[i, 0]), int(x2[i, 1]))
        cv2.line(img, pt1, pt2, color, 1)
        cv2.circle(img, pt1, 3, color, -1)
        cv2.circle(img, pt2, 3, color, -1)

    cv2.imshow("Matches Visualization", img)
    cv2.waitKey(0)

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    
    # Normalize errors for visualization
    err_img_init = cv2.normalize(err_img_init, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    err_img_optim = cv2.normalize(err_img_optim, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    err_img_init = cv2.applyColorMap(err_img_init, cv2.COLORMAP_JET)
    err_img_optim = cv2.applyColorMap(err_img_optim, cv2.COLORMAP_JET)

    # Helper to convert to BGR if needed
    def to_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    template_bgr = to_bgr(template)
    init_warp_bgr = to_bgr(img_warped_init)
    optim_warp_bgr = to_bgr(img_warped_optim)
    overlay_init_bgr = to_bgr(overlay_init)
    overlay_optim_bgr = to_bgr(overlay_optim)

    # Stack images
    row1 = np.hstack((template_bgr, init_warp_bgr, overlay_init_bgr, err_img_init))
    row2 = np.hstack((template_bgr, optim_warp_bgr, overlay_optim_bgr, err_img_optim))
    grid = np.vstack((row1, row2))

    # Resize to fit screen
    scale = 1200 / grid.shape[1]
    if scale < 1:
        grid = cv2.resize(grid, None, fx=scale, fy=scale)

    cv2.imshow("Alignment Results", grid)
    cv2.waitKey(0)

def warp_image(img, A, output_size):
    # To do
    '''
    This function performs Inverse (Backward) Warping. It transforms the input image img into a new
    image of size output_size based on the affine transformation matrix A.
    '''
    return cv2.warpAffine(img, A[:2, :], (output_size[1], output_size[0]))

if __name__ == "__main__":
    task4()
# captureImages.py
# Erik Sarkinen
# April 13th, 2026

import cv2
import os

# Create folder to save images
folder = 'checkerboard_images'
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize USB camera (index 0 is usually the default)
cap = cv2.VideoCapture(1)       # Left

# Configure resolution (optional, adjust to your camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count = 0
print("Press 's' to save a picture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live feed
    cv2.imshow('Camera Feed', frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        # Save image
        filename = os.path.join(folder, f'image_{count}.png')
        cv2.imwrite(filename, frame)
        print(f'Saved: {filename}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

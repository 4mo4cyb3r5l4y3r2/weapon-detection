import numpy as np
import cv2
import imutils
import datetime

# Load the trained cascade classifier
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Open the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

gun_exist = False  # Track whether a gun is detected

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame
    guns = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    # If any guns are detected, draw rectangles and mark existence
    if len(guns) > 0:
        gun_exist = True

    for (x, y, w, h) in guns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the current frame
    cv2.imshow('Security Feed', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Print final result
if gun_exist:
    print("Guns detected.")
else:
    print("No guns detected.")

# Release resources
camera.release()
cv2.destroyAllWindows()

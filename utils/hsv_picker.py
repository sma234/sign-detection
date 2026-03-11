"""
Utility script for tuning HSV color thresholds.
Click anywhere on the live frame to print the HSV value at that pixel.
"""
import cv2
import numpy as np


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV:", param["hsv"][y, x])


cap = cv2.VideoCapture(0)
print("Click on the image to sample HSV values. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Picker", frame)
    cv2.setMouseCallback("HSV Picker", on_click, param={"hsv": hsv})

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

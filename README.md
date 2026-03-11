# sign-detection
# Vroom: 1:10 Scale Autonomous Car - Traffic Sign Recognition

An end-to-end Computer Vision and Deep Learning pipeline designed to detect and classify traffic signs in real-time for a 1:10 scale autonomous vehicle (Vroom). This project demonstrates a complete edge AI workflow, from data preparation and model training to real-time deployment on an Nvidia Jetson.

## Key Highlights & Skills Demonstrated

* **Deep Learning (PyTorch):** Designed and trained a custom Convolutional Neural Network (`SmallSignNet`) from scratch to classify 10 different traffic sign categories.
* **Computer Vision (OpenCV):** Implemented a robust Region of Interest (ROI) extraction pipeline using HSV color space thresholding, morphological operations, and contour detection for real-time sign localization.
* **Edge AI Deployment:** Optimized the inference pipeline to run efficiently on resource-constrained hardware (Nvidia Jetson) using a live camera feed.
* **End-to-End Development:** Built the entire stack, including a custom PyTorch Dataset loader, training scripts, and an HSV calibration tool.

---

## Tech Stack

* **Language:** Python 3.x
* **Machine Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (cv2)
* **Data Manipulation:** NumPy, Pillow

---

## Pipeline Architecture

1. **Detection (Classic CV):** The system captures frames via the webcam and applies a Gaussian blur. It then uses carefully calibrated HSV masks (Red, Blue, Green, Yellow) to isolate potential traffic signs, followed by morphological operations to clean the masks and contour area/circularity filtering to extract accurate bounding boxes.
2. **Classification (Deep Learning / Template Matching):** The extracted ROIs are resized to 32x32 pixels and passed to the classification module. The project supports both classic Template Matching (NCC) and a trained PyTorch CNN for high-accuracy predictions.

---

## Setup & Installation

To set up the project on your local machine or Jetson device, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone 

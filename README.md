# Autonomous Edge AI Traffic Sign Recognition

An end-to-end Computer Vision and Deep Learning pipeline designed to detect and classify traffic signs in real-time. This project demonstrates a complete edge AI workflow, from data preparation and model training to real-time deployment on resource-constrained hardware like the Nvidia Jetson.

## Key Highlights & Skills Demonstrated

* **Deep Learning (PyTorch):** Designed and trained a custom Convolutional Neural Network (`SmallSignNet`) from scratch to classify 10 different traffic sign categories.
* **Computer Vision (OpenCV):** Implemented a robust Region of Interest (ROI) extraction pipeline using HSV color space thresholding, morphological operations, and contour detection for real-time sign localization.
* **Edge AI Deployment:** Optimized the inference pipeline to run efficiently on hardware using a live camera feed.
* **End-to-End Development:** Built the entire stack, including a custom PyTorch Dataset loader, training scripts, and an HSV calibration tool.

---

## Setup

To set up the project on your machine, you can follow these steps:

1. Set up git on your machine if not already configured.
2. Navigate to the directory you want to set up the project. Open a terminal in that folder and clone the project using git:
   ```bash
   git clone [https://github.com/sma234/sign-detection.git](https://github.com/sma234/sign-detection.git)
3. If you have VS Code installed, you can open the project from the same terminal using the following commands:
   ```bash
   cd sign-detection
   code .
4. Open the integrated VS Code terminal (via View > Terminal or by using the shortcut `Ctrl+``) and run the following commands to create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch torchvision opencv-python numpy pillow





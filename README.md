# 🛡️ Posture-Guard AI:

An intelligent computer vision agent built to eliminate desk-related back strain using dynamic machine learning profiling. Powered by Python, OpenCV, and the MediaPipe framework.

`Python 3.11+` `OpenCV` `MediaPipe Pose` `NumPy` 

---

### 🧩 The Problem
Standard computer-vision posture monitors rely on flawed, hardcoded geometric pixel thresholds. They constantly trigger false positives based on camera angle, user height, or seating distance, resulting in "alert fatigue" where corporate users simply uninstall the application. 

### 🚀 The Solution
PostureGuard AI abandons hardcoded geometry. Instead, it utilizes **Dynamic State Classification** (Euclidean distance matching). Through a brief setup phase, the AI learns the user's bespoke skeletal profile and adapts instantly to their unique desk setup, eliminating false positives and scaling dynamically.

---

### ✨ Key Features

* 🎯 **Dynamic Profile Classification:** Calculates the Euclidean distance between a user's live skeletal feed and their pre-recorded "Good" and "Bad" posture profiles for highly accurate, bespoke tracking.
* 📏 **Perspective Distortion Correction:** Automatically scales distance metrics using real-time eye-width anchoring. If the user moves closer to the webcam, the math dynamically adjusts to prevent perspective distortion from triggering a false slouch.
* 👁️ **Eye-Strain Protection:** Actively monitors the Z-axis proximity ratio. If the user's face breaches the 1.3x width threshold relative to their shoulders, the system triggers a "Too Close To Screen" warning to prevent ocular strain.
* 🔄 **False-Positive Prevention:** Separates neck rotation from spinal compression. If the facial width ratio drops drastically (indicating the user simply turned their head), the system categorizes it as "Neck Movement" rather than logging a posture failure.

---

### 🛠️ Tech Stack

* **Backend System:** Python 3.11
* **Computer Vision Framework:** OpenCV (`cv2`) for real-time webcam frame processing and RGB conversion.
* **Machine Learning Integration:** Google MediaPipe Pose (for high-speed, 33-point skeletal landmark extraction).
* **Mathematics & Matrices:** NumPy (for vector arrays and Euclidean distance calculations) and Python `math`.

---

### 💻 Local Setup Instructions

**1. Clone the repository:**
```bash
git clone [https://github.com/ishita230105/Posture-Guard.git](https://github.com/ishita230105/Posture-Guard.git)
cd Posture-Guard

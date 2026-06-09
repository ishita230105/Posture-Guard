# PostureGuard AI

An intelligent desktop application utilizing computer vision and multi-threaded processing architectures to monitor spinal alignment and minimize digital eye strain. 

---

### Engineering Challenges

Traditional computer-vision posture monitors rely heavily on fixed geometric thresholds. These configurations introduce severe vulnerabilities to perspective shifts, variations in user height, and seating depth fluctuations. The resulting baseline drift causes frequent false alerts, leading to alert fatigue where users eventually disable the software.

### The Architectural Solution

PostureGuard AI replaces static pixel boundaries with dynamic state classification via Euclidean distance matching ($L2$ Norm). A brief configuration stage allows the application to map a personalized skeletal coordinate structure that adjusts to the user's specific environmental geometry.

---

### Technical Architecture

The application isolates the heavy video processing pipeline from the interface layout engine to ensure performance stability:

* **Asynchronous Multi-Threading:** The machine learning inference engine (`MediaPipe Pose`) runs on a background worker thread. This keeps the core Tkinter UI loop running smoothly at 30+ FPS, preventing frame drops or interface freezes.
* **Temporal Noise Filtering:** Implements a 3.0-second time-window smoothing buffer. Short-term physical movements, such as reaching for objects or stretching, are automatically debounced to prevent notification spikes.

---

### Operational Features

* **Dynamic State Classification:** Calculates the Euclidean distance between incoming real-time spatial vectors and pre-recorded user baseline matrices for highly precise tracking.
* **Perspective Distortion Correction:** Normalizes calculated geometric lengths by dividing them by the live shoulder-width pixel footprint. This technique cancels out depth variations along the $Z$-axis.
* **Ocular Proximity Monitoring:** Tracks screen distance changes relative to the initial setup. If the user's facial geometry breaks a 1.3x width threshold relative to the shoulders, a warning triggers to prevent digital eye strain.
* **False-Positive Isolation:** Decouples pure neck rotation from spinal slouching. Sudden narrowing of facial vectors without vertical drops are marked as simple neck movements rather than posture failures.

---

### Component Stack

* **Core Execution Environment:** Python 3.11
* **User Interface:** Tkinter / TTK (Thread-safe scheduling using the built-in `.after()` method)
* **Computer Vision Engine:** OpenCV for hardware webcam access, color space conversions, and image serialization
* **Inference Layer:** Google MediaPipe Pose (33-point topological landmark extraction)
* **Data Processing:** NumPy for multidimensional array structures and vector math tracking

---

### Installation and Environment Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone [https://github.com/ishita230105/Posture-Guard.git](https://github.com/ishita230105/Posture-Guard.git)
   cd Posture-Guard
2. **Activate your Python virtual environment:**
   ```bash
   # Windows
   .\venv\Scripts\activate
3. **Install the optimized production dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the desktop engine:**
   ```bash
   python main.py
   

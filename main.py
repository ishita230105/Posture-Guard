import cv2
import mediapipe as mp
import math
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from plyer import notification

class PostureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("PostureGuard AI")
        self.window.geometry("950x550")
        self.window.configure(bg="#1e1e2e")
        
        # Core CV Variables
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        self.STATE_WAITING = 0
        self.STATE_MONITORING = 1
        self.current_state = self.STATE_WAITING
        
        self.good_profile = None
        self.bad_profile = None
        self.baseline_shldr_width = 0
        self.baseline_eye_ratio = 0
        
        self.slouch_start_time = None
        self.buffer_duration = 3.0  
        self.last_notification_time = 0
        self.notification_cooldown = 10.0
        
        self.is_running = True
        self.setup_ui()
        
        # Start background CV loop
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Custom styling styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 11, "bold"), pady=10, background="#89b4fa", foreground="#1e1e2e")
        style.map("TButton", background=[("active", "#b4befe")])
        
        # Left Side: Camera Canvas
        self.canvas = tk.Canvas(self.window, width=640, height=480, bg="#11111b", bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Right Side: Control Panel Frame
        self.control_frame = tk.Frame(self.window, bg="#1e1e2e")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_lbl = tk.Label(self.control_frame, text="PostureGuard AI", font=("Helvetica", 20, "bold"), bg="#1e1e2e", fg="#cdd6f4")
        title_lbl.pack(pady=(10, 20))
        
        # Status Readouts
        self.status_lbl = tk.Label(self.control_frame, text="SETUP MODE", font=("Helvetica", 14, "bold"), bg="#313244", fg="#f9e2af", width=22, pady=10)
        self.status_lbl.pack(pady=10)
        
        self.good_status_lbl = tk.Label(self.control_frame, text="[-] Good Profile Empty", font=("Helvetica", 10), bg="#1e1e2e", fg="#f38ba8")
        self.good_status_lbl.pack(pady=2)
        
        self.bad_status_lbl = tk.Label(self.control_frame, text="[-] Bad Profile Empty", font=("Helvetica", 10), bg="#1e1e2e", fg="#f38ba8")
        self.bad_status_lbl.pack(pady=2)
        
        # Control Buttons
        self.btn_good = ttk.Button(self.control_frame, text="Capture Good Posture", command=self.capture_good)
        self.btn_good.pack(fill=tk.X, pady=5)
        
        self.btn_bad = ttk.Button(self.control_frame, text="Capture Bad Posture", command=self.capture_bad)
        self.btn_bad.pack(fill=tk.X, pady=5)
        
        self.btn_start = ttk.Button(self.control_frame, text="Start Monitoring", command=self.start_monitoring, state=tk.DISABLED)
        self.btn_start.pack(fill=tk.X, pady=15)
        
        self.btn_reset = ttk.Button(self.control_frame, text="Reset Profile", command=self.reset_profiles)
        self.btn_reset.pack(fill=tk.X, pady=5)

    def extract_features(self, landmarks):
        l_shldr = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shldr = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]

        shldr_mid_y = (l_shldr.y + r_shldr.y) / 2
        ear_mid_y = (l_ear.y + r_ear.y) / 2
        shldr_width = math.dist([l_shldr.x, l_shldr.y], [r_shldr.x, r_shldr.y])
        
        if shldr_width == 0: 
            return None
        
        neck_length = abs(shldr_mid_y - ear_mid_y) / shldr_width
        head_drop = (nose.y - ear_mid_y) / shldr_width
        return np.array([neck_length, head_drop]), shldr_width, l_ear, r_ear, nose

    def send_alert(self, title, message):
        current_time = time.time()
        if current_time - self.last_notification_time > self.notification_cooldown:
            notification.notify(title=title, message=message, app_name="PostureGuard AI", timeout=3)
            self.last_notification_time = current_time

    def video_loop(self):
        while self.is_running:
            success, frame = self.cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb_frame)
            
            self.latest_features = None
            self.latest_shldr_width = 0
            self.latest_eye_ratio = 0
            
            if result.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                landmarks = result.pose_landmarks.landmark
                extracted = self.extract_features(landmarks)
                
                if extracted:
                    self.latest_features, self.latest_shldr_width, l_ear, r_ear, nose = extracted
                    l_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
                    r_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
                    
                    if self.latest_shldr_width > 0:
                        eye_width = math.dist([l_eye.x, l_eye.y], [r_eye.x, r_eye.y])
                        self.latest_eye_ratio = eye_width / self.latest_shldr_width

            if self.current_state == self.STATE_MONITORING and self.latest_features is not None:
                if self.latest_shldr_width > (self.baseline_shldr_width * 1.3):
                    self.update_status_ui("TOO CLOSE TO SCREEN!", "#fab387")
                    self.handle_alert_timer("Screen Proximity Risk")
                elif self.latest_eye_ratio < (self.baseline_eye_ratio * 0.65):
                    self.update_status_ui("NECK MOVEMENT", "#f9e2af")
                    self.handle_alert_timer("Neck Strain Risk")
                else:
                    dist_to_good = np.linalg.norm(self.latest_features - self.good_profile)
                    dist_to_bad = np.linalg.norm(self.latest_features - self.bad_profile)
                    
                    if dist_to_bad < dist_to_good:
                        self.update_status_ui("SLOUCH DETECTED!", "#f38ba8")
                        self.handle_alert_timer("Slouching Detected")
                    else:
                        self.update_status_ui("GREAT POSTURE", "#a6e3a1")
                        self.slouch_start_time = None

            # Render frame on Tkinter canvas
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            
            if self.is_running:
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk
                
            time.sleep(0.03)

    def handle_alert_timer(self, warning_type):
        if self.slouch_start_time is None:
            self.slouch_start_time = time.time()
        elif time.time() - self.slouch_start_time >= self.buffer_duration:
            self.send_alert("Posture Warning", f"Action required: {warning_type}")

    def update_status_ui(self, text, color):
        self.window.after(0, lambda: self.status_lbl.config(text=text, bg=color, fg="#1e1e2e"))

    def capture_good(self):
        if hasattr(self, 'latest_features') and self.latest_features is not None:
            self.good_profile = self.latest_features
            self.baseline_shldr_width = self.latest_shldr_width
            self.baseline_eye_ratio = self.latest_eye_ratio
            self.good_status_lbl.config(text="[x] Good Profile Saved", fg="#a6e3a1")
            self.check_ready_to_monitor()

    def capture_bad(self):
        if hasattr(self, 'latest_features') and self.latest_features is not None:
            self.bad_profile = self.latest_features
            self.bad_status_lbl.config(text="[x] Bad Profile Saved", fg="#a6e3a1")
            self.check_ready_to_monitor()

    def check_ready_to_monitor(self):
        if self.good_profile is not None and self.bad_profile is not None:
            self.btn_start.config(state=tk.NORMAL)

    def start_monitoring(self):
        self.current_state = self.STATE_MONITORING
        self.update_status_ui("MONITORING ACTIVE", "#a6e3a1")

    def reset_profiles(self):
        self.current_state = self.STATE_WAITING
        self.good_profile = None
        self.bad_profile = None
        self.baseline_shldr_width = 0
        self.baseline_eye_ratio = 0
        self.slouch_start_time = None
        self.good_status_lbl.config(text="[-] Good Profile Empty", fg="#f38ba8")
        self.bad_status_lbl.config(text="[-] Bad Profile Empty", fg="#f38ba8")
        self.btn_start.config(state=tk.DISABLED)
        self.update_status_ui("SETUP MODE", "#f9e2af")

    def on_close(self):
        self.is_running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()
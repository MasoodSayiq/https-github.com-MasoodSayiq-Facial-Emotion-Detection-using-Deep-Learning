# ui_app.py
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
import traceback
import datetime
import time
import threading
import queue

# =========================================================
# MODEL LOADER WITH FALLBACK (unchanged)
# =========================================================
def robust_load_model(candidate_paths):
    last_err = None
    for p in candidate_paths:
        try:
            if not os.path.exists(p):
                last_err = FileNotFoundError(f"Model file not found at: {p}")
                continue
            model = tf.keras.models.load_model(p, compile=False)
            print(f"Loaded model from: {p}")
            return model, p
        except Exception as e:
            last_err = e
            traceback.print_exc()
    try:
        for file in os.listdir("."):
            if file.endswith(".h5"):
                model = tf.keras.models.load_model(file, compile=False)
                print(f"Loaded fallback model from: {file}")
                return model, file
    except:
        pass
    raise RuntimeError(f"{last_err}\nTried: {candidate_paths}")

CANDIDATES = [
    r"D:\emotion_detection_project\data\models\best_model.h5",
    r"D:/emotion_detection_project/data/models/best_model.h5",
    r"..\data\models\best_model.h5",
    r"../data/Datasets/data/models/best_model.h5",
    "best_model.h5",
    "model.h5",
]

try:
    model, MODEL_PATH = robust_load_model(CANDIDATES)
except Exception as e:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Model Error", str(e))
    raise SystemExit(1)

# =========================================================
# CONSTANTS (unchanged)
# =========================================================
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': '#FF4757', 'Disgust': '#9C27B0', 'Fear': '#FF6348',
    'Happy': '#FFC107', 'Neutral': '#78909C', 'Sad': '#2196F3',
    'Surprise': '#FF9800'
}
EMOTION_EMOJIS = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Neutral': '😐', 'Sad': '😢',
    'Surprise': '😲'
}

# =========================================================
# STATISTICS TRACKER (unchanged)
# =========================================================
class EmotionStats:
    def __init__(self):
        self.reset_stats()
    def reset_stats(self):
        self.emotion_counts = {emotion: 0 for emotion in EMOTIONS}
        self.start_time = time.time()
        self.last_emotion = "Neutral"
        self.confidence_history = []
    def update(self, emotion, confidence):
        self.emotion_counts[emotion] += 1
        self.last_emotion = emotion
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
    def get_session_duration(self):
        return time.time() - self.start_time
    def get_emotion_percentages(self):
        total = sum(self.emotion_counts.values())
        if total == 0:
            return {emotion: 0 for emotion in EMOTIONS}
        return {emotion: (count / total) * 100 for emotion, count in self.emotion_counts.items()}
    def get_average_confidence(self):
        if not self.confidence_history:
            return 0
        return sum(self.confidence_history) / len(self.confidence_history)

# =========================================================
# MAIN UI APPLICATION with threading (A1: both capture & predict in threads)
# =========================================================
class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("AI Emotion Detection System")
        root.state("zoomed")
        root.configure(bg="#0A0E27")

        # Stats & model already loaded globally
        self.stats = EmotionStats()

        # Camera & threading primitives
        self.cap = None
        self.running = False
        self.video_writer = None

        # Thread coordination
        self.frame_queue = queue.Queue(maxsize=1)      # latest raw frame for prediction
        self.annot_queue = queue.Queue(maxsize=1)      # latest annotated frame for display
        self.predict_event = threading.Event()         # signal predictor
        self.stop_event = threading.Event()            # signal threads to stop
        self.capture_thread = None
        self.predict_thread = None

        # detection state
        self.current_emotion = "Neutral"
        self.confidence = 0.0

        # fixed camera resolution (as you used)
        self.camera_width = 640
        self.camera_height = 480

        # UI setup (keeps identical layout & look)
        self.setup_ui()

    # ---------------- UI (kept same as your layout) ----------------
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg="#0A0E27")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        header = tk.Frame(main_container, bg="#1A1F3A", height=100)
        header.pack(fill="x", pady=(0, 20))
        header.pack_propagate(False)
        title_frame = tk.Frame(header, bg="#1A1F3A")
        title_frame.pack(expand=True)
        tk.Label(title_frame, text="🎭", font=("Segoe UI", 48),
                 bg="#1A1F3A", fg="#00D9FF").pack(side="left", padx=10)
        tk.Label(title_frame, text="Emotion Detection System",
                 font=("Segoe UI", 36, "bold"),
                 bg="#1A1F3A", fg="#FFFFFF").pack(side="left")

        content = tk.Frame(main_container, bg="#0A0E27")
        content.pack(fill="both", expand=True)

        self.setup_camera_panel(content)
        self.setup_control_panel(content)

    def setup_camera_panel(self, parent):
        left_panel = tk.Frame(parent, bg="#1A1F3A", highlightthickness=2,
                             highlightbackground="#00D9FF")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 20))

        video_header = tk.Frame(left_panel, bg="#0F1629", height=60)
        video_header.pack(fill="x")
        video_header.pack_propagate(False)

        header_content = tk.Frame(video_header, bg="#0F1629")
        header_content.pack(expand=True)
        tk.Label(header_content, text="📹", font=("Segoe UI", 20),
                 bg="#0F1629", fg="#00D9FF").pack(side="left", padx=(20, 10))
        tk.Label(header_content, text="Live Camera Feed",
                 font=("Segoe UI", 16, "bold"),
                 bg="#0F1629", fg="#FFFFFF").pack(side="left")
        self.live_indicator = tk.Label(header_content, text="●",
                                       font=("Segoe UI", 20),
                                       bg="#0F1629", fg="#64748B")
        self.live_indicator.pack(side="left", padx=20)

        # fps label removed as requested

        self.video_frame = tk.Frame(left_panel, bg="#000000")
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame, bg="#0A0A0A",
                                    text="Camera Offline\n\n▶ Press Start Detection",
                                    font=("Segoe UI", 22), fg="#64748B",
                                    justify="center")
        self.video_label.pack(fill="both", expand=True)

    def setup_control_panel(self, parent):
        right_panel = tk.Frame(parent, bg="#1A1F3A", width=400,
                              highlightthickness=2, highlightbackground="#00D9FF")
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        self.configure_notebook_style()

        notebook = ttk.Notebook(right_panel, style="Custom.TNotebook")
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        detection_tab = tk.Frame(notebook, bg="#1A1F3A")
        notebook.add(detection_tab, text="🎯 Live Detection")
        self.setup_detection_tab(detection_tab)

        stats_tab = tk.Frame(notebook, bg="#10A6EC")
        notebook.add(stats_tab, text="📊 Statistics")
        self.setup_stats_tab(stats_tab)

    def configure_notebook_style(self):
        style = ttk.Style()
        style.configure("Custom.TNotebook", 
                       background="#FFFFFF",
                       borderwidth=0,
                       tabmargins=[0, 0, 0, 0])
        style.configure("Custom.TNotebook.Tab",
                       background="#0F1629",
                       foreground="#1B36E2",
                       padding=[20, 10],
                       font=("Segoe UI", 10, "bold"),
                       focuscolor="#1A1F3A")
        style.map("Custom.TNotebook.Tab",
                 background=[("selected", "#00D9FF"),
                           ("active", "#00B8D4")],
                 foreground=[("selected", "#0A0E27"),
                           ("active", "#0A0E27")])

    def _add_mousewheel_scroll(self, widget_canvas):
        def _on_mousewheel_windows(event):
            widget_canvas.yview_scroll(-int(event.delta / 120), "units")
        def _on_mousewheel_mac(event):
            widget_canvas.yview_scroll(-int(event.delta), "units")
        def _on_mousewheel_linux_up(event):
            widget_canvas.yview_scroll(-1, "units")
        def _on_mousewheel_linux_down(event):
            widget_canvas.yview_scroll(1, "units")
        widget_canvas.bind_all("<MouseWheel>", _on_mousewheel_windows, add="+")
        widget_canvas.bind_all("<Button-4>", _on_mousewheel_linux_up, add="+")
        widget_canvas.bind_all("<Button-5>", _on_mousewheel_linux_down, add="+")

    def setup_detection_tab(self, parent):
        container = tk.Frame(parent, bg="#1A1F3A")
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, bg="#1A1F3A", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        scroll_frame = tk.Frame(canvas, bg="#1A1F3A")
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind("<Configure>", on_frame_configure)
        def on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        self._add_mousewheel_scroll(canvas)

        emotion_card = tk.Frame(scroll_frame, bg="#0F1629", 
                               highlightthickness=1, highlightbackground="#00D9FF")
        emotion_card.pack(fill="x", padx=10, pady=10)
        self.emotion_display = tk.Label(emotion_card, text="😐",
                                        font=("Segoe UI", 80),
                                        bg="#0F1629", fg="#FFFFFF")
        self.emotion_display.pack(pady=15)
        self.emotion_text = tk.Label(emotion_card, text="Neutral",
                                     font=("Segoe UI", 24, "bold"),
                                     bg="#0F1629", fg="#78909C")
        self.emotion_text.pack(pady=(0, 15))

        conf_container = tk.Frame(scroll_frame, bg="#0F1629",
                                 highlightthickness=1, highlightbackground="#00D9FF")
        conf_container.pack(fill="x", padx=10, pady=10)
        tk.Label(conf_container, text="Confidence Level",
                 bg="#0F1629", fg="#94A3B8",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        self.conf_canvas = tk.Canvas(conf_container, height=30,
                                     bg="#1A1F3A", highlightthickness=0)
        self.conf_canvas.pack(fill="x", padx=20, pady=10)
        self.conf_label = tk.Label(conf_container, text="0%",
                                   font=("Segoe UI", 16, "bold"),
                                   bg="#0F1629", fg="#FFFFFF")
        self.conf_label.pack(pady=(0, 15))

        controls_frame = tk.Frame(scroll_frame, bg="#0F1629",
                                 highlightthickness=1, highlightbackground="#00D9FF")
        controls_frame.pack(fill="x", padx=10, pady=10)
        tk.Label(controls_frame, text="Controls",
                 bg="#0F1629", fg="#94A3B8",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20, pady=(15, 10))

        btn_container = tk.Frame(controls_frame, bg="#0F1629")
        btn_container.pack(fill="x", padx=20, pady=10)
        btn_style = {"font": ("Segoe UI", 12, "bold"), "bd": 0, "relief": "flat", "width": 20, "pady": 12, "cursor": "hand2"}

        self.start_btn = tk.Button(btn_container, text="▶ Start Detection",
                                   bg="#00D9FF", fg="#0A0E27",
                                   activebackground="#00B8D4",
                                   activeforeground="#0A0E27",
                                   command=self.start_camera, **btn_style)
        self.start_btn.pack(pady=8)
        self.stop_btn = tk.Button(btn_container, text="⏹ Stop Detection",
                                  bg="#FF4757", fg="white",
                                  activebackground="#E03E4D",
                                  activeforeground="white",
                                  state="disabled",
                                  command=self.stop_camera, **btn_style)
        self.stop_btn.pack(pady=8)
        tk.Button(btn_container, text="❌ Exit Application",
                  bg="#64748B", fg="white",
                  activebackground="#475569",
                  activeforeground="white",
                  command=self.exit_app, **btn_style).pack(pady=8)

    def setup_stats_tab(self, parent):
        stats_container = tk.Frame(parent, bg="#1A1F3A")
        stats_container.pack(fill="both", expand=True)
        canvas = tk.Canvas(stats_container, bg="#1A1F3A", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(stats_container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        scrollable_frame = tk.Frame(canvas, bg="#1A1F3A")
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        self._add_mousewheel_scroll(canvas)

        stats_frame = tk.Frame(scrollable_frame, bg="#0F1629", highlightthickness=1, highlightbackground="#00D9FF")
        stats_frame.pack(fill="x", padx=10, pady=10)
        tk.Label(stats_frame, text="Emotion Distribution", bg="#0F1629", fg="#94A3B8", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20, pady=(15, 10))

        self.stats_labels = {}
        stats_content = tk.Frame(stats_frame, bg="#0F1629")
        stats_content.pack(fill="x", padx=20, pady=10)
        for emotion in EMOTIONS:
            frame = tk.Frame(stats_content, bg="#0F1629")
            frame.pack(fill="x", pady=3)
            emoji_frame = tk.Frame(frame, bg="#0F1629")
            emoji_frame.pack(side="left", fill="x", expand=True)
            tk.Label(emoji_frame, text=EMOTION_EMOJIS[emotion], font=("Segoe UI", 14), bg="#0F1629", fg=EMOTION_COLORS[emotion]).pack(side="left", padx=(0, 10))
            tk.Label(emoji_frame, text=emotion, font=("Segoe UI", 10), bg="#0F1629", fg="#FFFFFF", width=12, anchor="w").pack(side="left")
            self.stats_labels[emotion] = tk.Label(frame, text="0%", font=("Segoe UI", 10, "bold"), bg="#0F1629", fg=EMOTION_COLORS[emotion])
            self.stats_labels[emotion].pack(side="right")
        separator = tk.Frame(stats_frame, height=2, bg="#00D9FF")
        separator.pack(fill="x", padx=20, pady=10)
        overall_frame = tk.Frame(stats_frame, bg="#0F1629")
        overall_frame.pack(fill="x", padx=20, pady=(10, 15))
        avg_frame = tk.Frame(overall_frame, bg="#0F1629")
        avg_frame.pack(fill="x", pady=2)
        tk.Label(avg_frame, text="Average Confidence:", font=("Segoe UI", 10), bg="#0F1629", fg="#94A3B8").pack(side="left")
        self.avg_conf_label = tk.Label(avg_frame, text="0%", font=("Segoe UI", 10, "bold"), bg="#0F1629", fg="#00D9FF")
        self.avg_conf_label.pack(side="right")
        duration_frame = tk.Frame(overall_frame, bg="#0F1629")
        duration_frame.pack(fill="x", pady=2)
        tk.Label(duration_frame, text="Session Duration:", font=("Segoe UI", 10), bg="#0F1629", fg="#94A3B8").pack(side="left")
        self.duration_label = tk.Label(duration_frame, text="00:00:00", font=("Segoe UI", 10, "bold"), bg="#0F1629", fg="#00D9FF")
        self.duration_label.pack(side="right")
        btn_frame = tk.Frame(stats_frame, bg="#0F1629")
        btn_frame.pack(fill="x", padx=20, pady=(10, 15))
        tk.Button(btn_frame, text="🔄 Reset Statistics", font=("Segoe UI", 10, "bold"), bg="#64748B", fg="white", activebackground="#475569", cursor="hand2", command=self.reset_statistics).pack(fill="x", pady=5)

    # ----------------- Stats & UI updater helpers -----------------
    def reset_statistics(self):
        self.stats.reset_stats()
        self.update_statistics_display()
        messagebox.showinfo("Statistics Reset", "All statistics have been reset.")

    def update_statistics_display(self):
        percentages = self.stats.get_emotion_percentages()
        for emotion in EMOTIONS:
            percentage = percentages[emotion]
            self.stats_labels[emotion].config(text=f"{percentage:.1f}%")
        avg_conf = self.stats.get_average_confidence()
        self.avg_conf_label.config(text=f"{avg_conf*100:.1f}%")
        duration = self.stats.get_session_duration()
        hours = int(duration // 3600); minutes = int((duration % 3600) // 60); seconds = int(duration % 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.duration_label.config(text=duration_str)

    # ----------------- Confidence bar -----------------
    def update_confidence_bar(self, conf):
        self.conf_canvas.delete("all")
        try:
            width = self.conf_canvas.winfo_width() or 350
        except:
            width = 350
        bar_w = int(width * conf)
        color = EMOTION_COLORS.get(self.current_emotion, "#00D9FF")
        self.conf_canvas.create_rectangle(0, 0, width, 30, fill="#1A1F3A", outline="")
        if bar_w > 0:
            self.conf_canvas.create_rectangle(0, 0, bar_w, 30, fill=color, outline="")
        self.conf_label.config(text=f"{int(conf*100)}%")

    # ================= Threading: capture & predict =================
    def start_camera(self):
        if self.running:
            return

        # Open camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access camera!")
            return

        # set resolution (kept same)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # prepare video writer (auto-save)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use AVI with XVID for better compatibility on Windows
        self.output_path = f"emotion_recording_{now}.avi"
        # widely supported codec on Windows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 20.0,
                                           (self.camera_width, self.camera_height), True)
        # verify writer opened successfully
        if not getattr(self.video_writer, "isOpened", lambda: False)():
            # Try releasing any partially created writer and disable saving
            try:
                self.video_writer.release()
            except:
                pass
            self.video_writer = None
            messagebox.showwarning("Recording Disabled", "Video recording could not be started (codec/backend not available). The app will still show live detection but will not save video.")
        else:
            print(f"VideoWriter opened, saving to: {self.output_path}")

        # reset thread control and stats
        self.stop_event.clear()
        self.predict_event.clear()
        # reset queues
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        with self.annot_queue.mutex:
            self.annot_queue.queue.clear()

        # set flags
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.live_indicator.config(fg="#FF4757")

        # spawn threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.predict_thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.capture_thread.start()
        self.predict_thread.start()

        # start the UI display polling loop
        self._schedule_display()

    def stop_camera(self):
        if not self.running:
            return
        # signal threads to stop
        self.running = False
        self.stop_event.set()
        # join threads safely
        try:
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            if self.predict_thread and self.predict_thread.is_alive():
                self.predict_thread.join(timeout=1.0)
        except Exception:
            pass

        # release camera & writer
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except:
            pass
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        except:
            pass

        # reset UI
        self.video_label.config(text="Camera Offline\n\n▶ Press Start Detection", image="")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.live_indicator.config(fg="#64748B")

        # show save message if video present
        if hasattr(self, 'output_path') and os.path.exists(self.output_path):
            messagebox.showinfo("Recording Saved", f"Video successfully saved to:\n{self.output_path}")

    # ---------------- capture thread ----------------
    def _capture_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            # write raw frame to video writer if exists and valid
            if self.video_writer:
                try:
                    # ensure frame size matches writer expectation
                    h, w = frame.shape[:2]
                    if (w, h) != (self.camera_width, self.camera_height):
                        # resize to expected size
                        frame_to_write = cv2.resize(frame, (self.camera_width, self.camera_height))
                    else:
                        frame_to_write = frame
                    self.video_writer.write(frame_to_write)
                except Exception as e:
                    # if writer fails, disable it so it doesn't keep failing
                    print(f"VideoWriter write error: {e}")
                    try:
                        self.video_writer.release()
                    except:
                        pass
                    self.video_writer = None

            # push latest raw frame for prediction (drop older)
            try:
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
                # signal predictor there's a new frame
                self.predict_event.set()
            except queue.Full:
                # if full, just continue (we always keep latest)
                pass

            # small sleep to yield CPU (camera read is main pacing)
            time.sleep(0.001)

    # ---------------- predict thread ----------------
    def _predict_loop(self):
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # Add confidence smoothing
        self.confidence_history = []
        self.confidence_smooth_window = 5
        
        while not self.stop_event.is_set():
            # wait until a frame is available or stop
            triggered = self.predict_event.wait(timeout=0.5)
            self.predict_event.clear()
            if self.stop_event.is_set():
                break
            try:
                frame = None
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    continue
                # operate on a copy
                proc_frame = frame.copy()
                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                max_conf = 0.0
                detected = "Neutral"

                for (x, y, w, h) in faces:
                    try:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (48, 48))
                        face_roi = face_roi.astype("float32") / 255.0
                        face_roi = np.expand_dims(face_roi, axis=(0, -1))

                        pred = model.predict(face_roi, verbose=0)
                        idx = np.argmax(pred)
                        conf = float(np.max(pred))
                        emotion = EMOTIONS[idx]

                        # Apply confidence smoothing
                        self.confidence_history.append(conf)
                        if len(self.confidence_history) > self.confidence_smooth_window:
                            self.confidence_history.pop(0)
                        smoothed_conf = sum(self.confidence_history) / len(self.confidence_history)

                        if smoothed_conf > max_conf:
                            max_conf = smoothed_conf
                            detected = emotion

                        color_hex = EMOTION_COLORS.get(emotion, "#00D9FF")
                        bgr = (int(color_hex[5:7], 16), int(color_hex[3:5], 16), int(color_hex[1:3], 16))

                        # draw bbox
                        cv2.rectangle(proc_frame, (x, y), (x+w, y+h), bgr, 3)
                        # emotion label top
                        emotion_label = f"{emotion}"
                        (emotion_w, emotion_h), baseline = cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(proc_frame, (x, y-emotion_h-10), (x+emotion_w+10, y), bgr, -1)
                        cv2.putText(proc_frame, emotion_label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        # confidence bottom - use smoothed confidence
                        confidence_label = f"{int(smoothed_conf*100)}%"
                        (conf_w, conf_h), _ = cv2.getTextSize(confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(proc_frame, (x, y+h), (x+conf_w+10, y+h+conf_h+10), bgr, -1)
                        cv2.putText(proc_frame, confidence_label, (x+5, y+h+conf_h+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Predict loop face error: {e}")
                        continue

                # update stats & UI-state with the best detected
                if max_conf > 0:
                    self.current_emotion = detected
                    self.confidence = max_conf
                    self.stats.update(detected, max_conf)
                    # push annotated frame for display (drop older)
                    try:
                        if not self.annot_queue.empty():
                            try:
                                self.annot_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.annot_queue.put_nowait(proc_frame)
                    except queue.Full:
                        pass
                else:
                    # even if no face, push simple frame to show (optional)
                    try:
                        if not self.annot_queue.empty():
                            try:
                                self.annot_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.annot_queue.put_nowait(proc_frame)
                    except queue.Full:
                        pass

            except Exception as e:
                print(f"Predict loop error: {e}")
                traceback.print_exc()
                time.sleep(0.01)
                continue

    # ---------------- UI display poller (main thread) ----------------
    def _schedule_display(self):
        # poll annotated frame and update UI elements
        try:
            frame = None
            try:
                frame = self.annot_queue.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                # convert and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # maintain aspect ratio to fit video_frame
                frame_width = self.video_frame.winfo_width()
                frame_height = self.video_frame.winfo_height()
                if frame_width > 10 and frame_height > 10:
                    img_ratio = pil_image.width / pil_image.height
                    frame_ratio = frame_width / frame_height
                    if img_ratio > frame_ratio:
                        new_width = frame_width
                        new_height = int(frame_width / img_ratio)
                    else:
                        new_height = frame_height
                        new_width = int(frame_height * img_ratio)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(pil_image)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk, text="")
            # Update UI widgets (emotion & confidence & stats)
            if self.current_emotion:
                self.emotion_display.config(text=EMOTION_EMOJIS.get(self.current_emotion, "😐"))
                self.emotion_text.config(text=self.current_emotion, fg=EMOTION_COLORS.get(self.current_emotion, "#FFFFFF"))
            if self.confidence is not None:
                self.update_confidence_bar(self.confidence)
            # update statistics display (not too frequently)
            self.update_statistics_display()
        except Exception as e:
            print(f"Display update error: {e}")
        # schedule next poll (10 ms)
        if self.running:
            self.root.after(10, self._schedule_display)

    # ---------------- exit ----------------
    def exit_app(self):
        if self.running:
            response = messagebox.askyesno("Confirm Exit", "Camera is still running. Exit anyway?")
            if not response:
                return
        # ensure threads stop and release
        try:
            self.stop_camera()
        except:
            pass
        self.root.destroy()

# =========================================================
# RUN APPLICATION
# =========================================================
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EmotionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()

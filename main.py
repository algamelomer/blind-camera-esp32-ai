import cv2
import requests
import subprocess
import threading
import time
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import platform

# ---- ESP32-CAM ----
STREAM_URL = "http://192.168.4.1:81/stream"
LED_URL = "http://192.168.4.1/led"

# ---- Load YOLOv8 ----
model = YOLO("yolov8n.pt")  # or yolov8n-int8 for speed

def set_led(state: bool, timeout=0.8):
    """Send LED on/off, mapped to your buzzer as well (GPIO4)."""
    try:
        r = requests.get(LED_URL, params={"state": "on" if state else "off"}, timeout=timeout)
        # Optional: check r.status_code == 200
    except requests.RequestException:
        pass  # network hiccup: ignore and keep going

# ---- TTS ----
def speak_text(text):
    """Use system TTS command for non-blocking speech"""
    try:
        if platform.system() == "Windows":
            # Windows - use PowerShell speech synthesis
            cmd = ["powershell", "-Command", f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"]
        elif platform.system() == "Darwin":  # macOS
            cmd = ["say", text]
        else:  # Linux
            cmd = ["espeak", text]
        
        # Run in background without waiting
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def say(phrase):
    """Speak phrase immediately without queue"""
    speak_text("I see " + phrase)
    print(f"Speaking: {phrase}")

# ---- Tkinter UI ----
root = tk.Tk()
root.title("Blind Assist Vision")
root.geometry("1200x700")
root.configure(bg="#121212")

# Title
title_frame = tk.Frame(root, bg="#121212")
title_frame.pack(pady=10)
logo = Image.open("logo.png").resize((80, 80))
logo_img = ImageTk.PhotoImage(logo)
tk.Label(title_frame, image=logo_img, bg="#121212").pack(side="left", padx=10)
tk.Label(title_frame, text="Blind Assist Vision", fg="white", bg="#121212",
         font=("Arial", 24, "bold")).pack(side="left")

# Video area
video_label = tk.Label(root, bg="black")
video_label.pack(pady=20)

# Detection log
log_box = tk.Text(root, height=10, width=100, bg="#1e1e1e", fg="white", font=("Consolas", 12))
log_box.pack(pady=10)

# Brightness indicator
brightness_label = tk.Label(root, text="Brightness: --", fg="white", bg="#121212", font=("Arial", 12))
brightness_label.pack(pady=5)

# Test TTS button
def test_tts():
    speak_text("Testing text to speech functionality")
    print("Test TTS triggered")

test_button = tk.Button(root, text="Test TTS", command=test_tts, bg="#4CAF50", fg="white", font=("Arial", 12))
test_button.pack(pady=5)

# ---- Global variables ----
frame_lock = threading.Lock()
latest_frame = None
last_spoken_objects = {}
last_tts_time = 0  # Global TTS cooldown
cap = cv2.VideoCapture(STREAM_URL)

# Check if camera is accessible
if not cap.isOpened():
    print(f"Warning: Could not open camera stream at {STREAM_URL}")
    print("Creating test pattern for development...")
    # Create a test pattern frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Camera Not Available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_frame, "Check ESP32-CAM connection", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    latest_frame = test_frame

# ---- Frame Capture Thread ----
def capture_frames():
    global latest_frame
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                time.sleep(1)  # Wait before retrying
                continue
            with frame_lock:
                latest_frame = frame
        except Exception as e:
            print(f"Frame capture error: {e}")
            time.sleep(1)  # Wait before retrying
            continue

threading.Thread(target=capture_frames, daemon=True).start()

# ---- Detection Thread ----
def detection_worker():
    global last_spoken_objects, latest_frame, last_tts_time
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)  # Wait a bit before checking again
                continue
            frame = latest_frame.copy()

        # Brightness check → control LED
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Update brightness indicator in UI
        brightness_label.config(text=f"Brightness: {brightness:.1f}")
        
        # Improved darkness detection with hysteresis
        # Turn on LED if dark (< 70), turn off if bright enough (> 90)
        # This prevents flickering around the threshold
        if brightness < 70:  # Dark - turn on lights
            set_led(True)  # Turn on LED
            print(f"Darkness detected: {brightness:.1f} - LED ON")
        elif brightness > 90:  # Bright enough - turn off lights
            set_led(False)  # Turn off LED
            print(f"Brightness detected: {brightness:.1f} - LED OFF")

        # Run YOLO
        try:
            results = model(frame, verbose=False)[0]
            h, w = frame.shape[:2]

            # Create a copy for drawing bounding boxes
            display_frame = frame.copy()

            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bw, bh = x2 - x1, y2 - y1

                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with background
                label_text = f"{label} {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Position
                if cx < w/3:
                    pos = "left"
                elif cx > 2*w/3:
                    pos = "right"
                else:
                    pos = "center"

                # Distance
                dist = "close" if bh > h/2 else "far"
                phrase = f"{label} {dist} {pos}"

                # Global TTS cooldown (1.5s) to prevent multiple speech at once
                now = time.time()
                if now - last_tts_time > 1.5:  # Only speak if enough time has passed
                    # Make phrase more unique by including position and distance
                    unique_phrase = f"{label}_{dist}_{pos}"
                    if unique_phrase not in last_spoken_objects or (now - last_spoken_objects[unique_phrase]) > 2.0:
                        print(f"Speaking detection: {phrase}")  # Debug output
                        say(phrase)
                        log_box.insert(tk.END, f"[INFO] {time.ctime()}: {phrase}\n")
                        log_box.see(tk.END)
                        last_spoken_objects[unique_phrase] = now
                        last_tts_time = now  # Update global TTS time
                    else:
                        print(f"Skipping {phrase} due to object cooldown ({(now - last_spoken_objects[unique_phrase]):.1f}s remaining)")
                else:
                    print(f"Skipping {phrase} due to global TTS cooldown ({(1.5 - (now - last_tts_time)):.1f}s remaining)")

            # Update the frame with bounding boxes
            with frame_lock:
                latest_frame = display_frame
                
                # Add brightness indicator to frame
                brightness_text = f"Brightness: {brightness:.1f}"
                if brightness < 70:
                    brightness_text += " (DARK - LED ON)"
                    color = (0, 255, 0)  # Green for LED on
                elif brightness > 90:
                    brightness_text += " (BRIGHT - LED OFF)"
                    color = (255, 255, 255)  # White for LED off
                else:
                    brightness_text += " (MEDIUM)"
                    color = (255, 255, 0)  # Yellow for medium
                
                cv2.putText(latest_frame, brightness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        except Exception as e:
            print(f"YOLO detection error: {e}")
            # Keep the original frame if detection fails
            with frame_lock:
                latest_frame = frame

        time.sleep(0.1)  # prevent CPU overload

threading.Thread(target=detection_worker, daemon=True).start()

# ---- UI Update ----
def update_ui():
    with frame_lock:
        if latest_frame is not None:
            frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((1000, 500))  # fit nicely
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

    root.after(30, update_ui)

update_ui()
root.mainloop()

# Cleanup
cap.release()

import os
import sys
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import time
import tkinter as tk
from tkinter import messagebox

# ========== CONFIG ==========
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
MATCH_THRESHOLD = 0.5
CAMERA_INDEX = 0
DISPLAY_DELAY = 5      # seconds to show face before exit
# ============================

# Detect mode (IN / OUT)
MODE = "IN"
if len(sys.argv) > 1 and sys.argv[1].upper() == "OUT":
    MODE = "OUT"

# Ensure folders/files exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "In_Time", "Out_Time"]).to_csv(ATTENDANCE_FILE, index=False)

# ========== LOAD KNOWN FACES ==========
print("[INFO] Loading known faces...")
known_faces, known_names = [], []
for fname in os.listdir(KNOWN_FACES_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    path = os.path.join(KNOWN_FACES_DIR, fname)
    name = os.path.splitext(fname)[0]
    image = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(image)
    if len(encs) > 0:
        known_faces.append(encs[0])
        known_names.append(name)
        print(f"[OK] Loaded: {name}")
if not known_faces:
    messagebox.showerror("Error", "No known faces found! Add images in 'known_faces' folder.")
    sys.exit()

# ========== LOAD EXISTING ATTENDANCE ==========
df = pd.read_csv(ATTENDANCE_FILE, dtype=str).fillna("")
today = datetime.now().strftime("%Y-%m-%d")

# ========== MARK ATTENDANCE ==========
def mark_attendance(name):
    global df
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    mask = (df["Name"] == name) & (df["Date"] == date_str)
    rows = df[mask]

    if MODE == "IN":
        if rows.empty:
            df = pd.concat([df, pd.DataFrame([{
                "Name": name, "Date": date_str, "In_Time": time_str, "Out_Time": ""
            }])], ignore_index=True)
            print(f"\n✅ {name} marked IN at {time_str}")
        else:
            print(f"\n[INFO] {name} already has IN today.")
    else:  # MODE == OUT
        if rows.empty:
            print(f"\n[INFO] No IN record found today for {name}; cannot mark OUT.")
        else:
            idx = rows.index[0]
            if str(df.at[idx, "Out_Time"]).strip() == "":
                df.at[idx, "Out_Time"] = time_str
                print(f"\n🚪 {name} marked OUT at {time_str}")
            else:
                print(f"\n[INFO] {name} already has OUT today.")

    df.to_csv(ATTENDANCE_FILE, index=False)

    # Pretty print
    print("\n📋 Attendance Records:")
    print("=" * 65)
    print(f"{'Name':<20}{'Date':<15}{'In_Time':<15}{'Out_Time':<15}")
    print("-" * 65)
    for _, r in df.iterrows():
        print(f"{r['Name']:<20}{r['Date']:<15}{r['In_Time']:<15}{r['Out_Time']:<15}")
    print("=" * 65)

# ========== CAMERA PROCESS ==========
def run_camera():
    video = cv2.VideoCapture(CAMERA_INDEX)
    if not video.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    print(f"\n[INFO] Camera started. Mode: {MODE}")
    recognized_name = None
    face_marked = False
    last_frame = None

    while True:
        ret, frame = video.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, face_locs)

        for enc, loc in zip(encs, face_locs):
            dists = face_recognition.face_distance(known_faces, enc)
            best_idx = np.argmin(dists)
            name_label = "Unknown"
            if dists[best_idx] < MATCH_THRESHOLD:
                name_label = known_names[best_idx]
                if not face_marked:
                    recognized_name = name_label
                    mark_attendance(name_label)
                    face_marked = True

            top, right, bottom, left = [v * 4 for v in loc]
            color = (0, 255, 0) if name_label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name_label, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            last_frame = frame.copy()

        cv2.imshow("Smart Attendance System", frame)

        if face_marked:
            print(f"\n[INFO] Recognized {recognized_name}. Keeping camera open for {DISPLAY_DELAY} seconds...")
            start = time.time()
            while time.time() - start < DISPLAY_DELAY:
                cv2.imshow("Smart Attendance System", last_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("[INFO] Program ended.")

# ========== GUI BUTTON WINDOW ==========
def open_out_mode():
    os.system(f'"{sys.executable}" "{__file__}" OUT')

root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("300x150")
root.resizable(False, False)

lbl = tk.Label(root, text=f"Current Mode: {MODE}", font=("Arial", 14))
lbl.pack(pady=10)

btn_start = tk.Button(root, text=f"Start {MODE} Recognition", font=("Arial", 12),
                      command=lambda: [root.destroy(), run_camera()])
btn_start.pack(pady=5)

btn_out = tk.Button(root, text="Mark OUT (Run Again)", font=("Arial", 12), bg="lightblue",
                    command=open_out_mode)
btn_out.pack(pady=5)

root.mainloop()
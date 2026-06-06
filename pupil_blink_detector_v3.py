"""
Pupil Size Change & Blink Frequency Detector  (v3 — updated for new MediaPipe)
===============================================================================
WHAT CHANGED FROM v2:
  - Updated to work with new MediaPipe versions (0.10.30+)
  - Uses mediapipe.tasks instead of mediapipe.solutions
  - Everything else (blink logic, pupil logic, overlay) is identical

Dependencies:
    pip3 install mediapipe opencv-python numpy requests

Usage:
    python3 pupil_blink_detector_v3.py
"""

import cv2 as cv
import numpy as np
import time
import urllib.request
import os
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# DOWNLOAD FACE LANDMARKER MODEL
# ─────────────────────────────────────────────

MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model... please wait")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BLINK_THRESHOLD       = 0.51
EYE_AR_CONSEC_FRAMES  = 2
BLINK_FREQ_WINDOW_SEC = 60.0

PUPIL_HISTORY_LEN     = 30
PUPIL_CHANGE_THRESH   = 1.5

# ─────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────

RIGHT_EYE_POINTS = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS  = [362, 385, 386, 387, 263, 373, 374, 380]

# ─────────────────────────────────────────────
# BLINK DETECTION
# ─────────────────────────────────────────────

def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3  - P13) ** 3
        + np.linalg.norm(P4  - P12) ** 3
        + np.linalg.norm(P5  - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    return numerator / denominator


def blinking_ratio(mesh_points_3D):
    right_ear = euclidean_distance_3D(mesh_points_3D[RIGHT_EYE_POINTS])
    left_ear  = euclidean_distance_3D(mesh_points_3D[LEFT_EYE_POINTS])
    return (right_ear + left_ear + 1) / 2


# ─────────────────────────────────────────────
# PUPIL DETECTION
# ─────────────────────────────────────────────

def get_pupil_radius(frame, mesh_points_2D, eye_landmarks):
    pts = mesh_points_2D[eye_landmarks]
    x, y, w, h = cv.boundingRect(pts)

    pad = 10
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, frame.shape[1])
    y2 = min(y + h + pad, frame.shape[0])

    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None, None, None

    gray    = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv.threshold(blurred, 45, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    pupil_contour = max(contours, key=cv.contourArea)
    (cx, cy), radius = cv.minEnclosingCircle(pupil_contour)

    real_cx = int(x1 + cx)
    real_cy = int(y1 + cy)

    return real_cx, real_cy, radius


# ─────────────────────────────────────────────
# PUPIL CHANGE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_pupil_change(history, threshold=PUPIL_CHANGE_THRESH):
    if len(history) < 2:
        return "Stable"
    current  = history[-1]
    baseline = float(np.mean(list(history)[:-1]))
    delta    = current - baseline
    if delta > threshold:
        return "Increasing"
    elif delta < -threshold:
        return "Decreasing"
    else:
        return "Stable"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # set up the new mediapipe face landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    total_blinks         = 0
    blink_frame_counter  = 0
    blink_timestamps     = deque()
    left_radius_history  = deque(maxlen=PUPIL_HISTORY_LEN)
    right_radius_history = deque(maxlen=PUPIL_HISTORY_LEN)

    print("Running... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        now = time.time()

        # convert frame for new mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # run detection
        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # 2D pixel coords
            mesh_2D = np.array(
                [[int(p.x * img_w), int(p.y * img_h)] for p in lm]
            )

            # 3D normalised coords for EAR
            mesh_3D = np.array([[p.x, p.y, p.z] for p in lm])

            # ── BLINK ──────────────────────────────
            ear = blinking_ratio(mesh_3D)

            if ear <= BLINK_THRESHOLD:
                blink_frame_counter += 1
            else:
                if blink_frame_counter > EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                    blink_timestamps.append(now)
                blink_frame_counter = 0

            while blink_timestamps and (now - blink_timestamps[0]) > BLINK_FREQ_WINDOW_SEC:
                blink_timestamps.popleft()

            blinks_per_minute = len(blink_timestamps) * (60.0 / BLINK_FREQ_WINDOW_SEC)

            # ── PUPIL SIZE ─────────────────────────
            l_result = get_pupil_radius(frame, mesh_2D, LEFT_EYE_POINTS)
            r_result = get_pupil_radius(frame, mesh_2D, RIGHT_EYE_POINTS)

            if None in l_result or None in r_result:
                cv.imshow("Pupil & Blink Detector", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            l_cx, l_cy, l_radius = l_result
            r_cx, r_cy, r_radius = r_result

            left_radius_history.append(l_radius)
            right_radius_history.append(r_radius)

            left_trend  = classify_pupil_change(left_radius_history)
            right_trend = classify_pupil_change(right_radius_history)
            avg_radius  = (l_radius + r_radius) / 2

            # draw circles around detected pupils
            cv.circle(frame, (l_cx, l_cy), int(l_radius), (0, 255, 255), 2, cv.LINE_AA)
            cv.circle(frame, (r_cx, r_cy), int(r_radius), (0, 255, 255), 2, cv.LINE_AA)

            # ── OVERLAY ────────────────────────────
            trend_colors = {
                "Increasing": (0, 255, 0),
                "Decreasing": (0, 0, 255),
                "Stable":     (200, 200, 200),
            }

            def put(text, y, color=(0, 255, 0)):
                cv.putText(frame, text, (20, y),
                           cv.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv.LINE_AA)

            put(f"EAR: {ear:.3f}",                      40)
            put(f"Total Blinks: {total_blinks}",         70)
            put(f"Blinks/min : {blinks_per_minute:.1f}", 100)
            put(f"L Pupil radius: {l_radius:.1f} px",   140)
            put(f"L Pupil trend : {left_trend}",         170, trend_colors[left_trend])
            put(f"R Pupil radius: {r_radius:.1f} px",   200)
            put(f"R Pupil trend : {right_trend}",        230, trend_colors[right_trend])
            put(f"Avg radius   : {avg_radius:.1f} px",  260)

            if blink_frame_counter > 0:
                cv.putText(frame, "BLINK", (img_w // 2 - 60, img_h - 30),
                           cv.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Pupil & Blink Detector", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"Session ended. Total blinks detected: {total_blinks}")


if __name__ == "__main__":
    main()

"""
Pupil Size Change Detector
==========================
Detects real pupil size (increase / decrease / stable)
using a normal webcam and MediaPipe face landmarks.

Dependencies:
    pip3 install mediapipe opencv-python-headless numpy

Usage:
    python3 pupil_size_detector.py
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
# DOWNLOAD MODEL
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

PUPIL_HISTORY_LEN   = 30    # how many frames to keep in history
PUPIL_CHANGE_THRESH = 1.5   # pixel change needed to call it increasing/decreasing

# ─────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────

RIGHT_EYE_POINTS = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS  = [362, 385, 386, 387, 263, 373, 374, 380]

# ─────────────────────────────────────────────
# PUPIL DETECTION
# ─────────────────────────────────────────────

def get_pupil_radius(frame, mesh_points_2D, eye_landmarks):
    """
    Finds the actual dark pupil blob inside the eye region.

    Steps:
      1. Crop just the eye area from the full frame
      2. Convert to grayscale
      3. Blur to reduce noise
      4. Threshold — anything darker than 45 = pupil
      5. Find the biggest dark blob = pupil
      6. Fit a circle around it and return the radius
    """
    pts = mesh_points_2D[eye_landmarks]
    x, y, w, h = cv.boundingRect(pts)

    # padding so we don't crop too tight
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

    # pupil is the darkest region in the eye
    _, thresh = cv.threshold(blurred, 45, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # biggest dark blob = pupil
    pupil_contour = max(contours, key=cv.contourArea)
    (cx, cy), radius = cv.minEnclosingCircle(pupil_contour)

    # convert back to full frame coordinates
    real_cx = int(x1 + cx)
    real_cy = int(y1 + cy)

    return real_cx, real_cy, radius


# ─────────────────────────────────────────────
# PUPIL CHANGE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_pupil_change(history, threshold=PUPIL_CHANGE_THRESH):
    """
    Compares current radius to the average of the last 30 frames.
    Returns: 'Increasing', 'Decreasing', or 'Stable'
    """
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
    # set up mediapipe
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

    left_radius_history  = deque(maxlen=PUPIL_HISTORY_LEN)
    right_radius_history = deque(maxlen=PUPIL_HISTORY_LEN)

    print("Running... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]

        # convert frame for mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # 2D pixel coordinates
            mesh_2D = np.array(
                [[int(p.x * img_w), int(p.y * img_h)] for p in lm]
            )

            # ── PUPIL DETECTION ────────────────────
            l_result = get_pupil_radius(frame, mesh_2D, LEFT_EYE_POINTS)
            r_result = get_pupil_radius(frame, mesh_2D, RIGHT_EYE_POINTS)

            # skip frame if detection failed
            if None in l_result or None in r_result:
                cv.imshow("Pupil Size Detector", frame)
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

            # draw yellow circles around detected pupils
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

            put(f"L Pupil radius: {l_radius:.1f} px",  40)
            put(f"L Pupil trend : {left_trend}",        70,  trend_colors[left_trend])
            put(f"R Pupil radius: {r_radius:.1f} px",  110)
            put(f"R Pupil trend : {right_trend}",       140, trend_colors[right_trend])
            put(f"Avg radius    : {avg_radius:.1f} px", 180)

        cv.imshow("Pupil Size Detector", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()

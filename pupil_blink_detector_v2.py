"""
Pupil Size Change & Blink Frequency Detector  (v2 — real pupil tracking)
=========================================================================
WHAT CHANGED FROM v1:
  - get_iris_radius() replaced with get_pupil_radius()
  - Now detects the actual dark pupil blob using brightness thresholding
  - Everything else (blink logic, overlay, main loop) is identical

HOW PUPIL DETECTION WORKS:
  1. Use MediaPipe eye landmarks to crop just the eye region from the frame
  2. Convert that crop to grayscale
  3. Blur slightly to reduce noise
  4. Threshold: anything darker than brightness 45 becomes black (the pupil)
  5. Find the biggest dark blob — that is the pupil
  6. Fit a circle around it and return the radius

LIMITATION:
  Still measured in pixels, not mm.
  Lighting changes will affect the reading — keep room lighting stable.

Dependencies:
    pip install opencv-python mediapipe numpy

Usage:
    python pupil_blink_detector_v2.py
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from collections import deque

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BLINK_THRESHOLD       = 0.51
EYE_AR_CONSEC_FRAMES  = 2
BLINK_FREQ_WINDOW_SEC = 60.0

PUPIL_HISTORY_LEN     = 30
PUPIL_CHANGE_THRESH   = 1.5

MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE  = 0.8

# ─────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────

RIGHT_EYE_POINTS = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS  = [362, 385, 386, 387, 263, 373, 374, 380]

# ─────────────────────────────────────────────
# BLINK DETECTION  (unchanged from v1)
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
# PUPIL DETECTION  (NEW — replaces iris method)
# ─────────────────────────────────────────────

def get_pupil_radius(frame, mesh_points_2D, eye_landmarks):
    """
    Detects the actual pupil (dark blob) inside the eye region.

    Args:
        frame           : the full BGR camera frame
        mesh_points_2D  : (N, 2) array of landmark pixel coordinates
        eye_landmarks   : list of landmark indices that surround the eye

    Returns:
        (center_x, center_y, radius) in full-frame pixel coordinates
        or (None, None, None) if detection failed this frame
    """
    pts = mesh_points_2D[eye_landmarks]
    x, y, w, h = cv.boundingRect(pts)

    # add padding so we don't crop too tight
    pad = 10
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, frame.shape[1])
    y2 = min(y + h + pad, frame.shape[0])

    eye_crop = frame[y1:y2, x1:x2]
    if eye_crop.size == 0:
        return None, None, None

    # convert to grayscale and blur to reduce noise
    gray    = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    # threshold: pupil is the darkest region
    # pixels darker than 45 → black (pupil), rest → white
    _, thresh = cv.threshold(blurred, 45, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # biggest dark blob = pupil
    pupil_contour = max(contours, key=cv.contourArea)
    (cx, cy), radius = cv.minEnclosingCircle(pupil_contour)

    # convert crop-relative coords back to full frame coords
    real_cx = int(x1 + cx)
    real_cy = int(y1 + cy)

    return real_cx, real_cy, radius


# ─────────────────────────────────────────────
# PUPIL CHANGE CLASSIFICATION  (unchanged)
# ─────────────────────────────────────────────

def classify_pupil_change(history: deque, threshold: float = PUPIL_CHANGE_THRESH):
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
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    total_blinks         = 0
    blink_frame_counter  = 0
    blink_timestamps     = deque()
    left_radius_history  = deque(maxlen=PUPIL_HISTORY_LEN)
    right_radius_history = deque(maxlen=PUPIL_HISTORY_LEN)

    print("Running… press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb_frame    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results      = mp_face_mesh.process(rgb_frame)
        now          = time.time()

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            mesh_2D = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in lm]
            )
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

            # ── PUPIL SIZE (real dark-blob method) ─
            l_result = get_pupil_radius(frame, mesh_2D, LEFT_EYE_POINTS)
            r_result = get_pupil_radius(frame, mesh_2D, RIGHT_EYE_POINTS)

            # skip frame if detection failed
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

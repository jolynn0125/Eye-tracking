"""
Eye Tracker — Group 5
=====================
Tracks gaze XY, blink frequency, and pupil size via webcam.

Displays on screen:
  - Gaze XY (average iris centre)
  - Accumulated blink count
  - Pupil mesh (iris circle) + avg radius + trend

Logs to CSV (per-second rows) on quit:
  timestamp | total_blinks | blink_freq_10s | gaze_x | gaze_y |
  gaze_displacement | pupil_size | pupil_trend

Usage:
    python eye_tracker.py
    python eye_tracker.py --name Alice
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import argparse
import time
import csv
import os
from datetime import datetime
from collections import deque

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_WEBCAM           = 0
LOG_FOLDER               = "logs"
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE  = 0.8

# Blink detection
BLINK_THRESHOLD          = 0.51   # EAR threshold (lower = more closed)
EYE_AR_CONSEC_FRAMES     = 2      # consecutive frames below threshold = blink

# Pupil detection
PUPIL_HISTORY_LEN        = 30     # rolling history length (frames)
PUPIL_CHANGE_THRESH      = 1.5   # px delta to call increasing/decreasing
PUPIL_DARK_THRESH        = 45     # grayscale threshold for dark pupil blob

# ─────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────

LEFT_EYE_IRIS   = [474, 475, 476, 477]
RIGHT_EYE_IRIS  = [469, 470, 471, 472]

# EAR landmarks (8 points per eye, 3-D)
RIGHT_EYE_EAR   = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_EAR    = [362, 385, 386, 387, 263, 373, 374, 380]

# Pupil crop landmarks (bounding box around eye)
LEFT_EYE_CROP   = [362, 385, 386, 387, 263, 373, 374, 380]
RIGHT_EYE_CROP  = [33,  160, 159, 158, 133, 153, 145, 144]

# ─────────────────────────────────────────────
# BLINK HELPERS
# ─────────────────────────────────────────────

def euclidean_distance_3D(points):
    """3-D Eye Aspect Ratio (EAR)."""
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    return numerator / denominator


def blinking_ratio(landmarks_3d):
    """Combined EAR for both eyes. Lower = more closed."""
    right = euclidean_distance_3D(landmarks_3d[RIGHT_EYE_EAR])
    left  = euclidean_distance_3D(landmarks_3d[LEFT_EYE_EAR])
    return (right + left + 1) / 2

# ─────────────────────────────────────────────
# PUPIL HELPERS
# ─────────────────────────────────────────────

def get_pupil_radius(frame, mesh_2d, eye_landmarks):
    """
    Detect the dark pupil blob inside the eye region.
    Returns (cx, cy, radius) in full-frame coordinates, or (None, None, None).
    """
    pts = mesh_2d[eye_landmarks]
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
    _, thresh = cv.threshold(blurred, PUPIL_DARK_THRESH, 255, cv.THRESH_BINARY_INV)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    pupil_contour     = max(contours, key=cv.contourArea)
    (cx, cy), radius  = cv.minEnclosingCircle(pupil_contour)

    return int(x1 + cx), int(y1 + cy), radius


def classify_pupil_change(history):
    """
    Compare latest radius to rolling mean of previous frames.
    Returns 'Increasing', 'Decreasing', or 'Stable'.
    """
    if len(history) < 2:
        return "Stable"
    current  = history[-1]
    baseline = float(np.mean(list(history)[:-1]))
    delta    = current - baseline
    if delta > PUPIL_CHANGE_THRESH:
        return "Increasing"
    elif delta < -PUPIL_CHANGE_THRESH:
        return "Decreasing"
    return "Stable"

# ─────────────────────────────────────────────
# OVERLAY HELPER
# ─────────────────────────────────────────────

def put(frame, text, y, color=(0, 255, 0)):
    cv.putText(frame, text, (20, y),
               cv.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv.LINE_AA)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eye Tracker — Group 5")
    parser.add_argument("--name", default="participant", help="Participant name for CSV filename")
    args = parser.parse_args()

    participant_name = args.name.strip().replace(" ", "_")

    # ── MediaPipe (solution API — same as code 1) ──────────────────────
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,          # needed for iris landmarks 468-477
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv.VideoCapture(DEFAULT_WEBCAM)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    # ── State ──────────────────────────────────────────────────────────
    total_blinks            = 0
    blink_frame_counter     = 0

    left_radius_history     = deque(maxlen=PUPIL_HISTORY_LEN)
    right_radius_history    = deque(maxlen=PUPIL_HISTORY_LEN)

    # Per-second / per-10s aggregation
    session_data            = []        # rows written to CSV
    second_blink_start      = 0         # blinks at the start of this second
    window_10s_blinks       = 0         # blinks in current 10-s window
    last_second_ts          = time.time()
    last_10s_ts             = time.time()
    prev_gaze               = None      # (x, y) from previous second

    trend_colors = {
        "Increasing": (0,  255, 0),
        "Decreasing": (0,  0,  255),
        "Stable":     (200, 200, 200),
    }

    print("Eye Tracker running. Press 'q' to quit and save CSV.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now       = time.time()
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results   = mp_face_mesh.process(rgb_frame)

            # ── Defaults when no face detected ─────────────────────────
            gaze_x, gaze_y   = None, None
            avg_radius        = None
            pupil_trend       = "Stable"

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]

                # 2-D pixel coords
                mesh_2d = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                     for p in lm.landmark]
                )
                # 3-D normalised (for EAR)
                mesh_3d = np.array([[n.x, n.y, n.z] for n in lm.landmark])

                # ── Blink detection ────────────────────────────────────
                ear = blinking_ratio(mesh_3d)
                if ear <= BLINK_THRESHOLD:
                    blink_frame_counter += 1
                else:
                    if blink_frame_counter > EYE_AR_CONSEC_FRAMES:
                        total_blinks += 1
                    blink_frame_counter = 0

                # ── Iris circles (gaze + pupil mesh) ──────────────────
                (l_cx, l_cy), l_r = cv.minEnclosingCircle(mesh_2d[LEFT_EYE_IRIS])
                (r_cx, r_cy), r_r = cv.minEnclosingCircle(mesh_2d[RIGHT_EYE_IRIS])

                # Draw iris circles (purple)
                cv.circle(frame, (int(l_cx), int(l_cy)), int(l_r), (255, 0, 255), 2, cv.LINE_AA)
                cv.circle(frame, (int(r_cx), int(r_cy)), int(r_r), (255, 0, 255), 2, cv.LINE_AA)

                # Average gaze position
                gaze_x = int((l_cx + r_cx) / 2)
                gaze_y = int((l_cy + r_cy) / 2)

                # ── Pupil size ─────────────────────────────────────────
                l_res = get_pupil_radius(frame, mesh_2d, LEFT_EYE_CROP)
                r_res = get_pupil_radius(frame, mesh_2d, RIGHT_EYE_CROP)

                if (
                    l_res is not None and
                    r_res is not None and
                    l_res[2] is not None and
                    r_res[2] is not None
                ):
                    _, _, l_radius = l_res
                    _, _, r_radius = r_res
                    
                    left_radius_history.append(l_radius)
                    right_radius_history.append(r_radius)

                    avg_radius = (l_radius + r_radius) / 2

                    pupil_trend = classify_pupil_change(
                        deque(
                            [(a + b) / 2
                            for a, b in zip(left_radius_history, right_radius_history)],
                            maxlen=PUPIL_HISTORY_LEN,
                        )
                    )


                # ── On-screen overlay ──────────────────────────────────
                put(frame, f"Blinks : {total_blinks}", 40)

                if gaze_x is not None:
                    put(frame, f"Gaze   : ({gaze_x}, {gaze_y})", 75)

                if avg_radius is not None:
                    put(frame, f"Pupil  : {avg_radius:.1f} px", 110)
                    put(frame, f"Trend  : {pupil_trend}", 145,
                        trend_colors[pupil_trend])

            # ── Per-second CSV row ─────────────────────────────────────
            if now - last_second_ts >= 1.0:
                elapsed_10s = now - last_10s_ts

                blinks_this_second = total_blinks - second_blink_start
                second_blink_start = total_blinks
                window_10s_blinks += blinks_this_second

                # Average blink freq per 10-s window (resets every 10 s)
                if elapsed_10s >= 10.0:
                    blink_freq_10s  = window_10s_blinks / elapsed_10s
                    window_10s_blinks = 0
                    last_10s_ts     = now
                else:
                    blink_freq_10s  = window_10s_blinks / elapsed_10s if elapsed_10s > 0 else 0.0

                # Gaze displacement from previous second
                if gaze_x is not None and prev_gaze is not None:
                    displacement = round(
                        np.sqrt((gaze_x - prev_gaze[0])**2 + (gaze_y - prev_gaze[1])**2), 2
                    )
                else:
                    displacement = None

                if gaze_x is not None:
                    prev_gaze = (gaze_x, gaze_y)

                session_data.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    total_blinks,
                    round(blink_freq_10s, 3),
                    gaze_x,
                    gaze_y,
                    displacement,
                    round(avg_radius, 2) if avg_radius is not None else None,
                    pupil_trend,
                ])

                last_second_ts = now

            # ── Display ────────────────────────────────────────────────
            cv.imshow("Eye Tracker — Group 5", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received.")
                break

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        cap.release()
        cv.destroyAllWindows()
        mp_face_mesh.close()

        # ── Save CSV ───────────────────────────────────────────────────
        if session_data:
            os.makedirs(LOG_FOLDER, exist_ok=True)
            ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{participant_name}_{ts}.csv"
            path     = os.path.join(LOG_FOLDER, filename)

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "total_blinks",
                    "blink_freq_10s",       # avg blinks/s in the current 10-s window; resets every 10s
                    "gaze_x",
                    "gaze_y",
                    "gaze_displacement",    # Euclidean distance from previous second's gaze
                    "pupil_size_px",
                    "pupil_trend",          # Increasing / Decreasing / Stable
                ])
                writer.writerows(session_data)

            print(f"Saved: {path}")
        else:
            print("No data recorded — CSV not saved.")

        print("Done.")


if __name__ == "__main__":
    main()
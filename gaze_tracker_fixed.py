"""
Gaze Tracker + Pupil Size Detector (Combined)
==============================================
Merges:
  - Gaze Tracker (iris tracking, gaze offsets, blink detection, face direction, UDP, CSV)
  - Pupil Size Detector (FaceLandmarker-based pupil radius + trend detection)

Both MediaPipe backends run in parallel on the same frame:
  • FaceMesh         → gaze / iris / blink / face direction
  • FaceLandmarker   → pupil size & trend (requires face_landmarker.task model)

The FaceLandmarker model is downloaded automatically on first run.

Controls:
  r  - start / stop recording to CSV
  q  - quit

"""

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socket
import argparse
import time
import csv
import urllib.request
from datetime import datetime
from collections import deque
import os

# ---------------------------------------------------------------------------
# Download FaceLandmarker model (required by pupil detector)
# ---------------------------------------------------------------------------

MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model... please wait")
    url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
           "face_landmarker/float16/1/face_landmarker.task")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

PRINT_DATA            = True
DEFAULT_WEBCAM        = 0
SHOW_ALL_FEATURES     = True   # overlay all 468 landmarks
SHOW_ON_SCREEN_DATA   = True
LOG_DATA              = True
LOG_FOLDER            = "logs"

SERVER_IP             = "127.0.0.1"
SERVER_PORT           = 7070
SERVER_ADDRESS        = (SERVER_IP, SERVER_PORT)

# Blink detection (gaze tracker)
TOTAL_BLINKS             = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD          = 0.51
EYE_AR_CONSEC_FRAMES     = 2

IS_RECORDING = False  # toggle with 'r'

# Pupil detection
PUPIL_HISTORY_LEN   = 30
PUPIL_CHANGE_THRESH = 1.5

# MediaPipe FaceMesh confidence
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE  = 0.8

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------

LEFT_EYE_IRIS          = [474, 475, 476, 477]
RIGHT_EYE_IRIS         = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER  = [33]
LEFT_EYE_INNER_CORNER  = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS       = [33,  160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS        = [362, 385, 386, 387, 263, 373, 374, 380]

# Face points for head-direction angles
_indices_pose   = [1, 33, 61, 199, 263, 291]
threshold_angle = 10

# ---------------------------------------------------------------------------
# Helper functions — Gaze Tracker
# ---------------------------------------------------------------------------

def vector_position(point1, point2):
    """Return (dx, dy) from point1 to point2."""
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1


def euclidean_distance_3D(points):
    """Eye Aspect Ratio in 3-D (used for blink detection)."""
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    return numerator / denominator


def blinking_ratio(landmarks):
    """Return combined EAR for both eyes (lower → more closed)."""
    right_ear = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_ear  = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    return (right_ear + left_ear + 1) / 2


# ---------------------------------------------------------------------------
# Helper functions — Pupil Size Detector
# ---------------------------------------------------------------------------

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

    pad = 10
    x1 = max(x - pad, 0);  y1 = max(y - pad, 0)
    x2 = min(x + w + pad, frame.shape[1]);  y2 = min(y + h + pad, frame.shape[0])

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


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Gaze Tracker + Pupil Size Detector")
parser.add_argument("-c", "--camSource", default=str(DEFAULT_WEBCAM),
                    help="Camera source index")
args = parser.parse_args()

if PRINT_DATA:
    print("Initialising face mesh and camera...")

# FaceMesh — used by gaze tracker
# FaceMesh replaced with FaceLandmarker (fixes mediapipe 0.10.30+ crash)
_base   = python.BaseOptions(model_asset_path=MODEL_PATH)
_opts   = vision.FaceLandmarkerOptions(
    base_options=_base,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
mp_face_mesh = vision.FaceLandmarker.create_from_options(_opts)

# FaceLandmarker — used by pupil detector
base_options    = python.BaseOptions(model_asset_path=MODEL_PATH)
fl_options      = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(fl_options)

cap         = cv.VideoCapture(int(args.camSource))
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Pupil radius history for trend detection
left_radius_history  = deque(maxlen=PUPIL_HISTORY_LEN)
right_radius_history = deque(maxlen=PUPIL_HISTORY_LEN)

# CSV setup
csv_data     = []
column_names = [
    "Timestamp (ms)",
    "Left Eye Center X",  "Left Eye Center Y",
    "Right Eye Center X", "Right Eye Center Y",
    "Left Iris Relative Pos Dx",  "Left Iris Relative Pos Dy",
    "Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy",
    "Total Blink Count",
]

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # ── Run both detectors on the same frame ─────────────────────────
        _mp_img          = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        facemesh_results = mp_face_mesh.detect(_mp_img)
        mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        fl_result        = face_landmarker.detect(mp_image)

        # ════════════════════════════════════════════════════════════════
        # GAZE TRACKER  (FaceMesh)
        # ════════════════════════════════════════════════════════════════
        if facemesh_results.face_landmarks:
            # 2-D pixel landmarks
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                 for p in facemesh_results.face_landmarks[0]]
            )
            # 3-D normalised landmarks (for EAR)
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z]
                 for n in facemesh_results.face_landmarks[0]]
            )

            # ── Face direction (angle_x / angle_y via solvePnP) ──────────
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )

            focal_length = 1 * img_w
            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2],
                 [0, focal_length, img_w / 2],
                 [0, 0, 1]]
            )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            pts_2D = np.delete(head_pose_points_3D, 2, axis=1).astype(np.float64)
            pts_3D = head_pose_points_3D.astype(np.float64)

            success, rot_vec, trans_vec = cv.solvePnP(
                pts_3D, pts_2D, cam_matrix, dist_matrix
            )
            rotation_matrix, _ = cv.Rodrigues(rot_vec)
            angles, *_ = cv.RQDecomp3x3(rotation_matrix)

            angle_x = angles[0] * 360
            angle_y = angles[1] * 360

            if angle_y < -threshold_angle:
                face_looks = "Left"
            elif angle_y > threshold_angle:
                face_looks = "Right"
            elif angle_x < -threshold_angle:
                face_looks = "Down"
            elif angle_x > threshold_angle:
                face_looks = "Up"
            else:
                face_looks = "Forward"

            if PRINT_DATA:
                print(f"Face looks: {face_looks}")

            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            # ── Blink detection ──────────────────────────────────────────
            ears = blinking_ratio(mesh_points_3D)
            if ears <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0

            # ── Optional: show all landmarks ─────────────────────────────
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            # ── Iris centres ─────────────────────────────────────────────
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left  = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Draw irises and eye corners
            cv.circle(frame, center_left,  int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_EYE_INNER_CORNER][0],  3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_EYE_OUTER_CORNER][0],  3, (0, 255, 255),   -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255),   -1, cv.LINE_AA)

            # ── Relative gaze offsets ────────────────────────────────────
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER],  center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

            # ── Console output ───────────────────────────────────────────
            if PRINT_DATA:
                print(f"Total Blinks: {TOTAL_BLINKS}")
                print(f"Left  Eye Centre  X: {l_cx:.0f}  Y: {l_cy:.0f}")
                print(f"Right Eye Centre  X: {r_cx:.0f}  Y: {r_cy:.0f}")
                print(f"Left  Iris Offset Dx: {l_dx}  Dy: {l_dy}")
                print(f"Right Iris Offset Dx: {r_dx}  Dy: {r_dy}\n")

            # ── On-screen overlay ────────────────────────────────────────
            if SHOW_ON_SCREEN_DATA:
                if IS_RECORDING:
                    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # red dot = recording
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}",
                           (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

            # ── CSV logging ──────────────────────────────────────────────
            if LOG_DATA and IS_RECORDING:
                timestamp = int(time.time() * 1000)
                csv_data.append([
                    timestamp,
                    l_cx, l_cy, r_cx, r_cy,
                    l_dx, l_dy, r_dx, r_dy,
                    TOTAL_BLINKS,
                ])

            # ── UDP packet ───────────────────────────────────────────────
            timestamp = int(time.time() * 1000)
            packet = (
                np.array([timestamp], dtype=np.int64).tobytes()
                + np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32).tobytes()
            )
            iris_socket.sendto(packet, SERVER_ADDRESS)

            if PRINT_DATA:
                print(f"Sent UDP packet to {SERVER_ADDRESS}")

        # ════════════════════════════════════════════════════════════════
        # PUPIL SIZE DETECTOR  (FaceLandmarker)
        # ════════════════════════════════════════════════════════════════
        if fl_result.face_landmarks:
            lm = fl_result.face_landmarks[0]

            # 2D pixel coordinates from FaceLandmarker
            mesh_2D = np.array(
                [[int(p.x * img_w), int(p.y * img_h)] for p in lm]
            )

            # ── Pupil detection ──────────────────────────────────────────
            l_result = get_pupil_radius(frame, mesh_2D, LEFT_EYE_POINTS)
            r_result = get_pupil_radius(frame, mesh_2D, RIGHT_EYE_POINTS)

            if None not in l_result and None not in r_result:
                lp_cx, lp_cy, lp_radius = l_result
                rp_cx, rp_cy, rp_radius = r_result

                left_radius_history.append(lp_radius)
                right_radius_history.append(rp_radius)

                left_trend  = classify_pupil_change(left_radius_history)
                right_trend = classify_pupil_change(right_radius_history)
                avg_radius  = (lp_radius + rp_radius) / 2

                # Draw yellow circles around detected pupils
                cv.circle(frame, (lp_cx, lp_cy), int(lp_radius), (0, 255, 255), 2, cv.LINE_AA)
                cv.circle(frame, (rp_cx, rp_cy), int(rp_radius), (0, 255, 255), 2, cv.LINE_AA)

                # ── Overlay ──────────────────────────────────────────────
                trend_colors = {
                    "Increasing": (0, 255, 0),
                    "Decreasing": (0, 0, 255),
                    "Stable":     (200, 200, 200),
                }

                def put(text, y, color=(0, 255, 0)):
                    cv.putText(frame, text, (20, y),
                               cv.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv.LINE_AA)

                put(f"L Pupil radius: {lp_radius:.1f} px",  110)
                put(f"L Pupil trend : {left_trend}",         140, trend_colors[left_trend])
                put(f"R Pupil radius: {rp_radius:.1f} px",  170)
                put(f"R Pupil trend : {right_trend}",        200, trend_colors[right_trend])
                put(f"Avg radius    : {avg_radius:.1f} px",  230)

                if PRINT_DATA:
                    print(f"L Pupil radius: {lp_radius:.1f} px  trend: {left_trend}")
                    print(f"R Pupil radius: {rp_radius:.1f} px  trend: {right_trend}")
                    print(f"Avg pupil radius: {avg_radius:.1f} px\n")

        # ── Display ──────────────────────────────────────────────────────
        cv.imshow("Gaze Tracker + Pupil Detector", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('r'):
            IS_RECORDING = not IS_RECORDING
            print("Recording", "started." if IS_RECORDING else "paused.")

        if key == ord('q'):
            if PRINT_DATA:
                print("Exiting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv.destroyAllWindows()
    iris_socket.close()

    if LOG_DATA and IS_RECORDING and csv_data:
        if PRINT_DATA:
            print("Writing CSV...")
        ts_str   = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_path = os.path.join(LOG_FOLDER, f"gaze_log_{ts_str}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            writer.writerows(csv_data)
        if PRINT_DATA:
            print(f"Data written to {csv_path}")

    if PRINT_DATA:
        print("Session ended.")

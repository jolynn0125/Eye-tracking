"""
Gaze Movement & Direction Detector
====================================
Detects:
  - Eye movement direction (Left / Right / Up / Down)
  - Whether eyes are moving (Saccade) or still (Fixation)
  - Overall activity level (engaged vs blank stare)

How it works:
  1. MediaPipe finds the iris center and eye corner landmarks
  2. Iris position is measured RELATIVE to the eye corners
     (so head movement doesn't count as eye movement)
  3. Every frame the iris position is compared to the last frame
  4. Large fast movement = Saccade (jumping between words)
  5. Small/no movement = Fixation (reading a word / focusing)

Dependencies:
    pip3 install mediapipe opencv-python-headless numpy

Usage:
    python3 gaze_movement_detector.py
"""

import cv2 as cv
import numpy as np
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

# How much the iris needs to move (in normalised units) to count as movement
# Lower = more sensitive, Higher = less sensitive
MOVEMENT_THRESHOLD  = 0.008

# How much movement to call it a Saccade (fast jump) vs slow drift
SACCADE_THRESHOLD   = 0.020

# How many frames of stillness = Fixation (reading/focusing)
FIXATION_MIN_FRAMES = 8

# Rolling history for average movement activity
ACTIVITY_HISTORY    = 60   # frames

# ─────────────────────────────────────────────
# LANDMARK INDICES
# ─────────────────────────────────────────────

# Iris centres (single centre point)
LEFT_IRIS_CENTER   = 468
RIGHT_IRIS_CENTER  = 473

# Eye corners — used to normalise iris position
LEFT_EYE_LEFT      = 263   # outer corner
LEFT_EYE_RIGHT     = 362   # inner corner
RIGHT_EYE_LEFT     = 133   # inner corner
RIGHT_EYE_RIGHT    = 33    # outer corner

# ─────────────────────────────────────────────
# GAZE LOGIC
# ─────────────────────────────────────────────

def get_relative_iris_position(lm, iris_center_idx,
                                eye_left_idx, eye_right_idx):
    """
    Returns iris position as a value between 0.0 and 1.0
    relative to the eye corners.
    0.0 = looking far left, 0.5 = centre, 1.0 = looking far right
    Also returns vertical position similarly.
    """
    iris  = lm[iris_center_idx]
    left  = lm[eye_left_idx]
    right = lm[eye_right_idx]

    eye_width  = abs(right.x - left.x)
    eye_height = abs(right.y - left.y) if abs(right.y - left.y) > 0.001 else 0.001

    # horizontal: 0 = left corner, 1 = right corner
    rel_x = (iris.x - min(left.x, right.x)) / eye_width if eye_width > 0 else 0.5

    # vertical: use iris y vs eye corner y midpoint
    mid_y = (left.y + right.y) / 2
    rel_y = iris.y - mid_y   # negative = looking up, positive = looking down

    return rel_x, rel_y


def classify_direction(dx, dy, threshold):
    """
    Given movement delta, return direction label.
    dx positive = moved right, negative = moved left
    dy positive = moved down, negative = moved up
    """
    if abs(dx) < threshold and abs(dy) < threshold:
        return "Still"

    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    else:
        return "Down" if dy > 0 else "Up"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
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

    # state
    prev_left_x,  prev_left_y  = None, None
    prev_right_x, prev_right_y = None, None

    fixation_counter  = 0
    activity_history  = deque(maxlen=ACTIVITY_HISTORY)

    total_saccades    = 0
    total_fixations   = 0

    print("Running... press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ── GET IRIS POSITIONS ─────────────────
            l_rel_x, l_rel_y = get_relative_iris_position(
                lm, LEFT_IRIS_CENTER, LEFT_EYE_LEFT, LEFT_EYE_RIGHT
            )
            r_rel_x, r_rel_y = get_relative_iris_position(
                lm, RIGHT_IRIS_CENTER, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT
            )

            # average both eyes together
            avg_x = (l_rel_x + r_rel_x) / 2
            avg_y = (l_rel_y + r_rel_y) / 2

            # ── MOVEMENT DETECTION ─────────────────
            direction  = "Still"
            move_type  = ""
            movement   = 0.0

            if prev_left_x is not None:
                dx = avg_x - ((prev_left_x + prev_right_x) / 2)
                dy = avg_y - ((prev_left_y + prev_right_y) / 2)
                movement = (dx**2 + dy**2) ** 0.5

                direction = classify_direction(dx, dy, MOVEMENT_THRESHOLD)

                if movement > SACCADE_THRESHOLD:
                    move_type = "Saccade"      # fast jump
                    fixation_counter = 0
                    total_saccades += 1
                elif movement > MOVEMENT_THRESHOLD:
                    move_type = "Moving"       # slow drift
                    fixation_counter = 0
                else:
                    fixation_counter += 1
                    if fixation_counter >= FIXATION_MIN_FRAMES:
                        move_type = "Fixation" # holding still = reading
                        if fixation_counter == FIXATION_MIN_FRAMES:
                            total_fixations += 1

                activity_history.append(movement)

            prev_left_x,  prev_left_y  = l_rel_x, l_rel_y
            prev_right_x, prev_right_y = r_rel_x, r_rel_y

            # ── ACTIVITY LEVEL ─────────────────────
            avg_activity = float(np.mean(activity_history)) if activity_history else 0
            if avg_activity > SACCADE_THRESHOLD * 0.5:
                activity_label = "Active (engaged)"
            elif avg_activity > MOVEMENT_THRESHOLD:
                activity_label = "Some movement"
            else:
                activity_label = "Still (blank stare?)"

            # ── DRAW IRIS DOTS ─────────────────────
            l_iris_px = (int(lm[LEFT_IRIS_CENTER].x  * img_w),
                         int(lm[LEFT_IRIS_CENTER].y  * img_h))
            r_iris_px = (int(lm[RIGHT_IRIS_CENTER].x * img_w),
                         int(lm[RIGHT_IRIS_CENTER].y * img_h))

            cv.circle(frame, l_iris_px, 5, (0, 255, 255), -1)
            cv.circle(frame, r_iris_px, 5, (0, 255, 255), -1)

            # ── DIRECTION ARROW ────────────────────
            arrow_map = {
                "Left":  "◀ Left",
                "Right": "▶ Right",
                "Up":    "▲ Up",
                "Down":  "▼ Down",
                "Still": "● Still",
            }
            arrow_label = arrow_map.get(direction, "")

            # ── OVERLAY ────────────────────────────
            move_colors = {
                "Saccade":  (0, 165, 255),   # orange
                "Moving":   (255, 255, 0),   # yellow
                "Fixation": (0, 255, 0),     # green
                "":         (200, 200, 200),
            }
            dir_color = (255, 255, 255)

            def put(text, y, color=(0, 255, 0)):
                cv.putText(frame, text, (20, y),
                           cv.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv.LINE_AA)

            put(f"Direction  : {arrow_label}",        40,  dir_color)
            put(f"Movement   : {move_type}",           70,  move_colors.get(move_type, (200,200,200)))
            put(f"Activity   : {activity_label}",     100,  (200, 200, 200))
            put(f"Saccades   : {total_saccades}",     140,  (0, 165, 255))
            put(f"Fixations  : {total_fixations}",    170,  (0, 255, 0))
            put(f"Movement px: {movement:.4f}",       200,  (200, 200, 200))

            # big direction label in corner
            cv.putText(frame, arrow_label,
                       (img_w - 200, img_h - 30),
                       cv.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Gaze Movement Detector", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"Session ended.")
    print(f"Total Saccades : {total_saccades}")
    print(f"Total Fixations: {total_fixations}")


if __name__ == "__main__":
    main()

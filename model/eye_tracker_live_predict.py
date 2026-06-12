"""
Eye Tracker + Live Daydreaming Prediction — Group 5

This script:
1. Opens the webcam.
2. Tracks gaze, blinking, and pupil size.
3. Aggregates the live data into 30-second windows.
4. Sends the 30-second feature sample to the trained model.
5. Displays and logs Focused/Daydreaming prediction.

Before running:
    python train_loso.py

Then run:
    python eye_tracker_live_predict.py --name TestUser
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import argparse
import time
import csv
import os
import joblib
import pandas as pd

from datetime import datetime
from collections import deque


DEFAULT_WEBCAM = 0
LOG_FOLDER = "logs"

MODEL_PATH = "daydreaming_detector_model.pkl"

MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

PREDICTION_WINDOW_SECONDS = 30

BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2

PUPIL_HISTORY_LEN = 30
PUPIL_CHANGE_THRESH = 1.5
PUPIL_DARK_THRESH = 45

LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]

RIGHT_EYE_EAR = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_EAR = [362, 385, 386, 387, 263, 373, 374, 380]

LEFT_EYE_CROP = [362, 385, 386, 387, 263, 373, 374, 380]
RIGHT_EYE_CROP = [33, 160, 159, 158, 133, 153, 145, 144]


def euclidean_distance_3d(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points

    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

    if denominator == 0:
        return 1.0

    return numerator / denominator


def blinking_ratio(landmarks_3d):
    right = euclidean_distance_3d(landmarks_3d[RIGHT_EYE_EAR])
    left = euclidean_distance_3d(landmarks_3d[LEFT_EYE_EAR])
    return (right + left + 1) / 2


def get_pupil_radius(frame, mesh_2d, eye_landmarks):
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

    gray = cv.cvtColor(eye_crop, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv.threshold(
        blurred,
        PUPIL_DARK_THRESH,
        255,
        cv.THRESH_BINARY_INV,
    )

    contours, _ = cv.findContours(
        thresh,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, None, None

    pupil_contour = max(contours, key=cv.contourArea)

    if cv.contourArea(pupil_contour) <= 0:
        return None, None, None

    (cx, cy), radius = cv.minEnclosingCircle(pupil_contour)

    if radius <= 0:
        return None, None, None

    return int(x1 + cx), int(y1 + cy), float(radius)


def classify_pupil_change(history):
    if len(history) < 2:
        return "Stable"

    current = history[-1]
    baseline = float(np.mean(list(history)[:-1]))
    delta = current - baseline

    if delta > PUPIL_CHANGE_THRESH:
        return "Increasing"

    if delta < -PUPIL_CHANGE_THRESH:
        return "Decreasing"

    return "Stable"


def put(frame, text, y, color=(0, 255, 0)):
    cv.putText(
        frame,
        text,
        (20, y),
        cv.FONT_HERSHEY_DUPLEX,
        0.65,
        color,
        2,
        cv.LINE_AA,
    )


def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run train_loso.py first to create the trained model."
        )

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle["model"]
        features = bundle["features"]
        label_mapping = bundle.get("label_mapping", {0: "Focused", 1: "Daydreaming"})
    else:
        model = bundle
        features = ["Mean_Pupil_Size", "Blink_Freq", "Gaze_Disp"]
        label_mapping = {0: "Focused", 1: "Daydreaming"}

    return model, features, label_mapping


def aggregate_prediction_window(rows):
    pupil_values = [
        row["pupil_size_px"]
        for row in rows
        if row["pupil_size_px"] is not None
    ]

    blink_values = [
        row["blink_freq_10s"]
        for row in rows
        if row["blink_freq_10s"] is not None
    ]

    gaze_values = [
        row["gaze_displacement"]
        for row in rows
        if row["gaze_displacement"] is not None
    ]

    if len(pupil_values) == 0 or len(blink_values) == 0 or len(gaze_values) == 0:
        return None

    return {
        "Mean_Pupil_Size": float(np.mean(pupil_values)),
        "Blink_Freq": float(np.mean(blink_values)),
        "Gaze_Disp": float(np.mean(gaze_values)),
    }


def predict_attention(model, feature_names, label_mapping, feature_sample):
    sample_df = pd.DataFrame([feature_sample])
    sample_df = sample_df[feature_names]

    raw_prediction = int(model.predict(sample_df)[0])
    prediction_label = label_mapping.get(raw_prediction, str(raw_prediction))

    focused_probability = None
    daydreaming_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample_df)[0]
        class_labels = list(model.classes_)

        if 0 in class_labels:
            focused_probability = float(probabilities[class_labels.index(0)])

        if 1 in class_labels:
            daydreaming_probability = float(probabilities[class_labels.index(1)])

    return {
        "prediction": prediction_label,
        "focused_probability": focused_probability,
        "daydreaming_probability": daydreaming_probability,
    }


def main():
    parser = argparse.ArgumentParser(description="Eye Tracker + Live Prediction")
    parser.add_argument("--name", default="participant", help="Participant name for CSV filename")
    parser.add_argument("--camera", type=int, default=DEFAULT_WEBCAM, help="Webcam index")
    args = parser.parse_args()

    participant_name = args.name.strip().replace(" ", "_")

    model, feature_names, label_mapping = load_model_bundle()

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Model features: {feature_names}")
    print("Live prediction starts after the first 30-second window.")
    print("Press 'q' to quit and save CSV.")

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Error: cannot open camera.")
        mp_face_mesh.close()
        return

    total_blinks = 0
    blink_frame_counter = 0

    left_radius_history = deque(maxlen=PUPIL_HISTORY_LEN)
    right_radius_history = deque(maxlen=PUPIL_HISTORY_LEN)

    session_data = []
    prediction_window_rows = []

    second_blink_start = 0
    window_10s_blinks = 0

    last_second_ts = time.time()
    last_10s_ts = time.time()
    prediction_window_start_ts = time.time()

    prev_gaze = None

    latest_prediction = "Waiting..."
    latest_focused_probability = None
    latest_daydreaming_probability = None

    trend_colors = {
        "Increasing": (0, 255, 0),
        "Decreasing": (0, 0, 255),
        "Stable": (200, 200, 200),
    }

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame from camera.")
                break

            now = time.time()
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            results = mp_face_mesh.process(rgb_frame)

            gaze_x = None
            gaze_y = None
            avg_radius = None
            pupil_trend = "Stable"

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]

                mesh_2d = np.array([
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in lm.landmark
                ])

                mesh_3d = np.array([
                    [p.x, p.y, p.z]
                    for p in lm.landmark
                ])

                ear = blinking_ratio(mesh_3d)

                if ear <= BLINK_THRESHOLD:
                    blink_frame_counter += 1
                else:
                    if blink_frame_counter >= EYE_AR_CONSEC_FRAMES:
                        total_blinks += 1

                    blink_frame_counter = 0

                (l_cx, l_cy), l_r = cv.minEnclosingCircle(mesh_2d[LEFT_EYE_IRIS])
                (r_cx, r_cy), r_r = cv.minEnclosingCircle(mesh_2d[RIGHT_EYE_IRIS])

                cv.circle(frame, (int(l_cx), int(l_cy)), int(l_r), (255, 0, 255), 2, cv.LINE_AA)
                cv.circle(frame, (int(r_cx), int(r_cy)), int(r_r), (255, 0, 255), 2, cv.LINE_AA)

                gaze_x = int((l_cx + r_cx) / 2)
                gaze_y = int((l_cy + r_cy) / 2)

                l_res = get_pupil_radius(frame, mesh_2d, LEFT_EYE_CROP)
                r_res = get_pupil_radius(frame, mesh_2d, RIGHT_EYE_CROP)

                if l_res[2] is not None and r_res[2] is not None:
                    l_radius = l_res[2]
                    r_radius = r_res[2]

                    left_radius_history.append(l_radius)
                    right_radius_history.append(r_radius)

                    avg_radius = (l_radius + r_radius) / 2

                    combined_history = deque(
                        [
                            (left + right) / 2
                            for left, right in zip(left_radius_history, right_radius_history)
                        ],
                        maxlen=PUPIL_HISTORY_LEN,
                    )

                    pupil_trend = classify_pupil_change(combined_history)

                put(frame, f"Blinks : {total_blinks}", 40)

                if gaze_x is not None:
                    put(frame, f"Gaze   : ({gaze_x}, {gaze_y})", 75)

                if avg_radius is not None:
                    put(frame, f"Pupil  : {avg_radius:.1f} px", 110)
                    put(frame, f"Trend  : {pupil_trend}", 145, trend_colors[pupil_trend])

            if now - last_second_ts >= 1.0:
                elapsed_10s = now - last_10s_ts

                blinks_this_second = total_blinks - second_blink_start
                second_blink_start = total_blinks
                window_10s_blinks += blinks_this_second

                if elapsed_10s >= 10.0:
                    blink_freq_10s = window_10s_blinks / elapsed_10s
                    window_10s_blinks = 0
                    last_10s_ts = now
                else:
                    blink_freq_10s = window_10s_blinks / elapsed_10s if elapsed_10s > 0 else 0.0

                if gaze_x is not None and prev_gaze is not None:
                    gaze_displacement = round(
                        np.sqrt((gaze_x - prev_gaze[0]) ** 2 + (gaze_y - prev_gaze[1]) ** 2),
                        2,
                    )
                else:
                    gaze_displacement = None

                if gaze_x is not None:
                    prev_gaze = (gaze_x, gaze_y)

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_blinks": total_blinks,
                    "blink_freq_10s": round(blink_freq_10s, 3),
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "gaze_displacement": gaze_displacement,
                    "pupil_size_px": round(avg_radius, 2) if avg_radius is not None else None,
                    "pupil_trend": pupil_trend,
                    "prediction": latest_prediction,
                    "focused_probability": latest_focused_probability,
                    "daydreaming_probability": latest_daydreaming_probability,
                }

                session_data.append(row)
                prediction_window_rows.append(row)

                last_second_ts = now

            if now - prediction_window_start_ts >= PREDICTION_WINDOW_SECONDS:
                feature_sample = aggregate_prediction_window(prediction_window_rows)

                if feature_sample is not None:
                    result = predict_attention(
                        model=model,
                        feature_names=feature_names,
                        label_mapping=label_mapping,
                        feature_sample=feature_sample,
                    )

                    latest_prediction = result["prediction"]
                    latest_focused_probability = result["focused_probability"]
                    latest_daydreaming_probability = result["daydreaming_probability"]

                    print("\n================ LIVE PREDICTION ================")
                    print("30s feature sample:", feature_sample)
                    print("Prediction:", latest_prediction)

                    if latest_focused_probability is not None:
                        print(f"Focused probability: {latest_focused_probability:.3f}")

                    if latest_daydreaming_probability is not None:
                        print(f"Daydreaming probability: {latest_daydreaming_probability:.3f}")

                else:
                    print("\nNot enough valid pupil/gaze/blink data for prediction.")

                prediction_window_rows = []
                prediction_window_start_ts = now

            if latest_prediction == "Daydreaming":
                prediction_color = (0, 0, 255)
            elif latest_prediction == "Focused":
                prediction_color = (0, 255, 0)
            else:
                prediction_color = (255, 255, 0)

            put(frame, f"Prediction: {latest_prediction}", 190, prediction_color)

            if latest_daydreaming_probability is not None:
                put(
                    frame,
                    f"Daydreaming Prob: {latest_daydreaming_probability:.2f}",
                    225,
                    prediction_color,
                )

            cv.imshow("Eye Tracker + Live Prediction", frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                print("Quit signal received.")
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        mp_face_mesh.close()

        if session_data:
            os.makedirs(LOG_FOLDER, exist_ok=True)

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{participant_name}_{ts}_with_predictions.csv"
            path = os.path.join(LOG_FOLDER, filename)

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "total_blinks",
                    "blink_freq_10s",
                    "gaze_x",
                    "gaze_y",
                    "gaze_displacement",
                    "pupil_size_px",
                    "pupil_trend",
                    "prediction",
                    "focused_probability",
                    "daydreaming_probability",
                ])

                for row in session_data:
                    writer.writerow([
                        row["timestamp"],
                        row["total_blinks"],
                        row["blink_freq_10s"],
                        row["gaze_x"],
                        row["gaze_y"],
                        row["gaze_displacement"],
                        row["pupil_size_px"],
                        row["pupil_trend"],
                        row["prediction"],
                        row["focused_probability"],
                        row["daydreaming_probability"],
                    ])

            print(f"Saved: {path}")
        else:
            print("No data recorded. CSV not saved.")

        print("Done.")


if __name__ == "__main__":
    main()
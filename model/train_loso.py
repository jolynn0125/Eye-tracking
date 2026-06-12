import os
import glob
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


DATA_DIR = "data"
OUTPUT_DIR = "outputs"

FEATURES = ["Mean_Pupil_Size", "Blink_Freq", "Gaze_Disp"]
TARGET = "Label"
GROUP_COL = "ID"

MODEL_OUTPUT_PATH = "daydreaming_detector_model.pkl"


def read_data_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file type: {file_path}")


def infer_participant_id_from_filename(file_path):
    filename = os.path.basename(file_path)

    # Expected examples:
    # P01_External.csv
    # P01_Internal.csv
    # P01_External - Sheet1.csv
    parts = filename.split("_")

    if parts and parts[0].upper().startswith("P"):
        return int(parts[0].upper().replace("P", ""))

    return None


def infer_label_from_filename(file_path):
    filename = os.path.basename(file_path).lower()

    if "external" in filename or "focused" in filename:
        return 0

    if "internal" in filename or "daydream" in filename or "daydreaming" in filename:
        return 1

    return None


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {
        "pupil_size_px": "Mean_Pupil_Size",
        "pupil_size": "Mean_Pupil_Size",
        "mean_pupil_size": "Mean_Pupil_Size",

        "blink_freq_10s": "Blink_Freq",
        "blink_freq": "Blink_Freq",
        "Blink_Frequency": "Blink_Freq",

        "gaze_displacement": "Gaze_Disp",
        "gaze_disp": "Gaze_Disp",
        "Gaze_Displacement": "Gaze_Disp",

        "label": "Label",
        "participant_id": "ID",
        "id": "ID",
    }

    df = df.rename(columns=rename_map)
    return df


def load_all_files(data_dir):
    all_files = []

    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        all_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not all_files:
        raise FileNotFoundError(f"No CSV/Excel files found in: {data_dir}")

    dataframes = []

    for file_path in sorted(all_files):
        df = read_data_file(file_path)
        df = normalize_columns(df)

        if "ID" not in df.columns:
            inferred_id = infer_participant_id_from_filename(file_path)

            if inferred_id is None:
                raise ValueError(
                    f"Missing ID column and could not infer participant ID from filename: {file_path}"
                )

            df["ID"] = inferred_id

        if "Label" not in df.columns:
            inferred_label = infer_label_from_filename(file_path)

            if inferred_label is None:
                raise ValueError(
                    f"Missing Label column and could not infer label from filename: {file_path}"
                )

            df["Label"] = inferred_label

        required_cols = [GROUP_COL, TARGET] + FEATURES
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"{file_path} is missing required columns: {missing}")

        df = df[required_cols].copy()
        df["source_file"] = os.path.basename(file_path)

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def clean_data(df):
    df = df.copy()

    for col in [GROUP_COL, TARGET] + FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[GROUP_COL, TARGET] + FEATURES)
    after = len(df)

    if before != after:
        print(f"Removed {before - after} rows with invalid/missing values.")

    df[GROUP_COL] = df[GROUP_COL].astype(int)
    df[TARGET] = df[TARGET].astype(int)

    invalid_labels = sorted(set(df[TARGET].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}. Expected only 0 and 1.")

    return df


def print_dataset_summary(df):
    print("\n================ DATASET SUMMARY ================")
    print(f"Total samples: {len(df)}")
    print(f"Participants: {sorted(df[GROUP_COL].unique())}")

    print("\nSamples per participant:")
    print(df.groupby(GROUP_COL)[TARGET].count())

    print("\nClass distribution:")
    print(df[TARGET].value_counts().rename(index={0: "Focused", 1: "Daydreaming"}))

    print("\nMean feature values by class:")
    print(
        df.groupby(TARGET)[FEATURES]
        .mean()
        .rename(index={0: "Focused", 1: "Daydreaming"})
    )


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ]),
    }


def run_loso_evaluation(df, feature_set):
    X = df[feature_set]
    y = df[TARGET]
    groups = df[GROUP_COL]

    logo = LeaveOneGroupOut()
    models = get_models()

    results = {}

    for model_name, model in models.items():
        print(f"\n\n================ {model_name} ================")

        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            test_participant = groups.iloc[test_idx].unique()[0]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            print(f"Test Participant P{test_participant:02d} Accuracy: {acc:.3f}")

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

        accuracy = accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(y_true_all, y_pred_all, zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

        print("\nOverall Results:")
        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1 Score : {f1:.3f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_true_all,
                y_pred_all,
                labels=[0, 1],
                target_names=["Focused", "Daydreaming"],
                zero_division=0,
            )
        )

    return results


def compare_feature_sets(df):
    feature_sets = {
        "All Features": ["Mean_Pupil_Size", "Blink_Freq", "Gaze_Disp"],
        "Only Pupil": ["Mean_Pupil_Size"],
        "Only Blink": ["Blink_Freq"],
        "Only Gaze": ["Gaze_Disp"],
        "Blink + Gaze": ["Blink_Freq", "Gaze_Disp"],
        "Pupil + Blink": ["Mean_Pupil_Size", "Blink_Freq"],
        "Pupil + Gaze": ["Mean_Pupil_Size", "Gaze_Disp"],
    }

    rows = []

    for feature_set_name, feature_set in feature_sets.items():
        print(f"\n\n################################################")
        print(f"FEATURE SET: {feature_set_name}")
        print(f"FEATURES: {feature_set}")
        print(f"################################################")

        results = run_loso_evaluation(df, feature_set)

        for model_name, result in results.items():
            rows.append({
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
            })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by="F1", ascending=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_feature_comparison.csv"), index=False)

    print("\n\n================ FEATURE COMPARISON ================")
    print(comparison_df)

    return comparison_df


def plot_confusion_matrices(results, title_suffix):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name, result in results.items():
        cm = result["confusion_matrix"]

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Focused", "Daydreaming"],
        )

        disp.plot()
        plt.title(f"{model_name} - {title_suffix}")
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"confusion_matrix_{model_name.replace(' ', '_').lower()}_{title_suffix.replace(' ', '_').lower()}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.show()


def plot_feature_comparison(comparison_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(12, 7))

    labels = comparison_df["Feature_Set"] + " | " + comparison_df["Model"]
    scores = comparison_df["F1"]

    plt.barh(labels, scores)
    plt.xlabel("F1 Score")
    plt.title("Feature Set and Model Comparison")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "feature_model_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_feature_distributions(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for feature in FEATURES:
        focused = df[df[TARGET] == 0][feature]
        daydreaming = df[df[TARGET] == 1][feature]

        plt.figure(figsize=(7, 5))
        plt.boxplot([focused, daydreaming], labels=["Focused", "Daydreaming"])
        plt.ylabel(feature)
        plt.title(f"{feature}: Focused vs Daydreaming")
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, f"boxplot_{feature}.png")
        plt.savefig(output_path, dpi=300)
        plt.show()


def train_final_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    final_model.fit(X, y)

    model_bundle = {
        "model": final_model,
        "features": FEATURES,
        "label_mapping": {
            0: "Focused",
            1: "Daydreaming",
        },
    }

    joblib.dump(model_bundle, MODEL_OUTPUT_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "features": FEATURES,
        "label_mapping": {
            "0": "Focused",
            "1": "Daydreaming",
        },
        "training_samples": int(len(df)),
        "participants": [int(x) for x in sorted(df[GROUP_COL].unique())],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nFinal model saved as: {MODEL_OUTPUT_PATH}")
    print(f"Model metadata saved as: {os.path.join(OUTPUT_DIR, 'model_metadata.json')}")


def main():
    df = load_all_files(DATA_DIR)
    df = clean_data(df)

    print_dataset_summary(df)

    print("\n\nRunning LOSO evaluation using all features...")
    all_feature_results = run_loso_evaluation(df, FEATURES)
    plot_confusion_matrices(all_feature_results, "All Features")

    print("\n\nComparing feature sets...")
    comparison_df = compare_feature_sets(df)
    plot_feature_comparison(comparison_df)
    plot_feature_distributions(df)

    print("\n\nTraining final model on all available labelled data...")
    train_final_model(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
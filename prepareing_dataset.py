from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("outputs/payload")
OUTPUT_DIR = Path("outputs/training/datasets")

WINDOW_SIZE = 15

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FEATURES = [
    "acc_x_mps2",
    "acc_y_mps2",
    "acc_z_mps2",
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",
]

LABELS = [
    "swing_angle_x_rad",
    "swing_angle_y_rad",
    "swing_rate_x_radps",
    "swing_rate_y_radps",
]

TIME_COLUMN = "time_s"


# ============================================================
# Load CSV files
# ============================================================

def load_data():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR.resolve()}"
        )

    print(f"Found {len(files)} CSV files.")

    required = [TIME_COLUMN] + FEATURES + LABELS
    loaded_data = []

    for file in files:
        print(f"\nLoading: {file.name}")

        df = pd.read_csv(file)

        original_rows = len(df)

        # Remove the initial hovering phase.
        if "trajectory_started" not in df.columns:
            raise ValueError(
                f"{file.name}: 'trajectory_stated' column not found. "
                "Please regenerate this CSV with the update simulation script."
            )

        df["trajectory_started"] = pd.to_numeric(
            df["trajectory_started"], errors="raise"
        )

        start_indices = np.flatnonzero(
            df["trajectory_started"].to_numpy() == 1
        )

        if len(start_indices) == 0:
            raise ValueError(
                f"{file.name}: No trajectory-start marker found."
            )

        start_idx = int(start_indices[0])
        strat_time = float(df.iloc[start_idx]["time_s"])

        df = df.iloc[start_idx:].copy()
        df = df.drop(columns=["trajectory_started"])
        df = df.reset_index(drop=True)

        removed_rows = original_rows - len(df)

        print(f"Original rows:              {original_rows}")
        print(f"Hovering rows removed:      {removed_rows}")
        print(f"Rows after hover removal:   {len(df)}")
        print(f"Trajectory start time:      {strat_time:.3f} s")

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"{file.name}: Missing required columns: {missing}"
            )

        df = df[required].copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df.isna().any().any():
            bad_counts = df.isna().sum()
            bad_counts = bad_counts[bad_counts > 0]
            raise ValueError(
                f"{file.name}: Missing or non-numeric values:\n"
                f"{bad_counts}"
            )

        values = df[required].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{file.name}: NaN or infinite values detected."
            )

        time = df[TIME_COLUMN].to_numpy(dtype=np.float64)

        if np.any(np.diff(time) <= 0):
            raise ValueError(
                f"{file.name}: Timestamps must increase strictly."
            )

        if len(df) <= WINDOW_SIZE + 2:
            print(f"Skipping {file.name}: not enough rows.")
            continue

        print(f"Rows: {len(df)}")
        print(f"Duration: {time[-1] - time[0]:.2f} s")
        print(f"Median dt: {np.median(np.diff(time)):.6f} s")

        loaded_data.append((file, df))

    if not loaded_data:
        raise ValueError("No usable CSV files were found.")

    return loaded_data


# ============================================================
# Create windows
# ============================================================

def make_windows(data, window_size):
    """
    Given N time samples, construct:

        X[i] = data[i : i + window_size, FEATURES]
        y[i] = data[i + window_size, LABELS]

    Thus, the label belongs to the time step immediately
    following the 15 input samples.
    """

    x_values = data[FEATURES].to_numpy(dtype=np.float32)
    y_values = data[LABELS].to_numpy(dtype=np.float32)

    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(x_values[i:i + window_size])
        y.append(y_values[i + window_size])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


# ============================================================
# Main preprocessing
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_data = load_data()

    # Split every flight log independently.
    # Windows will never cross flight or split boundaries.
    train_segments = []
    val_segments = []
    test_segments = []

    for source_file, df in loaded_data:
        n = len(df)

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        if min(len(train_df), len(val_df), len(test_df)) <= WINDOW_SIZE:
            raise ValueError(
                f"{source_file.name}: A split is too short. "
                "Use longer logs or adjust the split ratios."
            )

        train_segments.append(train_df)
        val_segments.append(val_df)
        test_segments.append(test_df)

        print(
            f"{source_file.name}: "
            f"train={len(train_df)}, "
            f"val={len(val_df)}, "
            f"test={len(test_df)}"
        )

    # Fit normalization parameters using training data only.
    train_features = np.concatenate(
        [
            segment[FEATURES].to_numpy(dtype=np.float64)
            for segment in train_segments
        ],
        axis=0,
    )

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    # Normalize each segment using training-set statistics.
    def normalize_segments(segments):
        normalized = []

        for segment in segments:
            segment = segment.copy()
            segment.loc[:, FEATURES] = (
                segment[FEATURES] - feature_mean
            ) / feature_std
            normalized.append(segment)

        return normalized

    train_segments = normalize_segments(train_segments)
    val_segments = normalize_segments(val_segments)
    test_segments = normalize_segments(test_segments)

    # Create windows independently for each segment.
    # No window can cross a flight-log or split boundary.
    def combine_windows(segments):
        all_X = []
        all_y = []

        for segment in segments:
            X_part, y_part = make_windows(segment, WINDOW_SIZE)

            if len(X_part) > 0:
                all_X.append(X_part)
                all_y.append(y_part)

        if not all_X:
            raise ValueError("No usable windows were created.")

        return (
            np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
        )

    X_train, y_train = combine_windows(train_segments)
    X_val, y_val = combine_windows(val_segments)
    X_test, y_test = combine_windows(test_segments)

    output_file = OUTPUT_DIR / "swing_dataset.npz"

    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
    )

    metadata = {
        "source_files": [str(file) for file, _ in loaded_data],
        "number_of_csv_files": len(loaded_data),
        "window_size": WINDOW_SIZE,
        "features": FEATURES,
        "labels": LABELS,
        "target_timing": (
            "The label is the sample immediately after "
            "the input window."
        ),
        "input_shape_per_sample": [
            WINDOW_SIZE,
            len(FEATURES),
        ],
        "output_shape_per_sample": [len(LABELS)],
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }

    metadata_file = OUTPUT_DIR / "metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDataset creation completed.")
    print(f"CSV files used: {len(loaded_data)}")
    print(f"Dataset: {output_file}")
    print(f"Metadata: {metadata_file}")

    print("\nWindowed dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")

if __name__ == "__main__":
    main()
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

    if len(files) > 1:
        print(
            f"WARNING: {len(files)} CSV files found. "
            "This initial version uses the most recently "
            "modified CSV only."
        )
        files = [
            max(files, key=lambda p: p.stat().st_mtime)
        ]

    file = files[0]
    df = pd.read_csv(file)

    print(f"Input file: {file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required = [TIME_COLUMN] + FEATURES + LABELS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Keep only the variables required for this experiment.
    df = df[required].copy()

    # Convert required columns to numeric values.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reject invalid rows rather than silently removing them.
    if df.isna().any().any():
        bad_counts = df.isna().sum()
        bad_counts = bad_counts[bad_counts > 0]

        raise ValueError(
            "Missing or non-numeric values detected:\n"
            f"{bad_counts}"
        )

    values = df[required].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "NaN or infinite values detected."
        )

    # Time must increase strictly.
    time = df[TIME_COLUMN].to_numpy(dtype=np.float64)
    dt = np.diff(time)

    if np.any(dt <= 0):
        raise ValueError(
            "Timestamps must be strictly increasing."
        )

    print(f"Start time: {time[0]:.6f} s")
    print(f"End time:   {time[-1]:.6f} s")
    print(f"Median dt:  {np.median(dt):.6f} s")
    print(f"Median sampling rate: "
          f"{1.0 / np.median(dt):.2f} Hz")

    print("\nInput statistics:")
    print(df[FEATURES].describe().loc[["mean", "std", "min", "max"]])

    print("\nSwing-angle statistics:")
    print(df[LABELS].describe().loc[["mean", "std", "min", "max"]])

    return file, df


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

    source_file, df = load_data()

    n = len(df)

    if n <= WINDOW_SIZE + 2:
        raise ValueError(
            "Not enough rows to create time windows."
        )

    # Split chronologically BEFORE creating windows.
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\nRaw time-series split:")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")

    # Fit standardization parameters on training inputs only.
    train_features = train_df[FEATURES].to_numpy(
        dtype=np.float64
    )

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)

    # Avoid division by zero for constant features.
    feature_std[feature_std < 1e-8] = 1.0

    # Apply the same transformation to all splits.
    for split_df in [train_df, val_df, test_df]:
        split_df.loc[:, FEATURES] = (
            split_df[FEATURES] - feature_mean
        ) / feature_std

    # Create each split independently so windows do not cross
    # train/validation/test boundaries.
    X_train, y_train = make_windows(
        train_df, WINDOW_SIZE
    )
    X_val, y_val = make_windows(
        val_df, WINDOW_SIZE
    )
    X_test, y_test = make_windows(
        test_df, WINDOW_SIZE
    )

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError(
            "At least one split has no usable windows."
        )

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
        "source_file": str(source_file),
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
    print(f"Dataset:  {output_file}")
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
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("outputs/training/datasets/swing_dataset.npz")

RESULTS_DIR = Path("outputs/training/results")

PREDICTION_DIR = RESULTS_DIR/"prediction"
LOSS_DIR = RESULTS_DIR/"loss"
METRICS_DIR = RESULTS_DIR/"metrics"

CHECKPOINT_DIR = Path("outputs/training/checkpoints")

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15

INPUT_SIZE = 6
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4


# ============================================================
# Reproducibility and device
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
LOSS_DIR.mkdir(parents=True,exist_ok=True)
METRICS_DIR.mkdir(parents=True,exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")


# ============================================================
# Model: LSTM + MLP
# ============================================================

class SwingLSTM(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        # x shape: (batch, time_steps, input_features)
        sequence_output, _ = self.lstm(x)

        # Use the final output of the input window.
        last_output = sequence_output[:, -1, :]

        # Predict the two swing angles at the target time.
        return self.mlp(last_output)


# ============================================================
# Load prepared dataset
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run the dataset preparation script first."
    )

with np.load(DATA_PATH) as data:
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    feature_mean = data["feature_mean"].astype(np.float32)
    feature_std = data["feature_std"].astype(np.float32)


def to_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader = to_loader(X_val, y_val, shuffle=False)
test_loader = to_loader(X_test, y_test, shuffle=False)

print("\nDataset shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")


# ============================================================
# Training setup
# ============================================================

model = SwingLSTM().to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

train_losses = []
val_losses = []

checkpoint_path = CHECKPOINT_DIR / "best_swing_lstm.pt"


# ============================================================
# Training loop
# ============================================================

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(xb)
        loss = criterion(predictions, yb)

        loss.backward()
        optimizer.step()

        train_sum += loss.item() * len(xb)
        train_count += len(xb)

    train_loss = train_sum / train_count

    model.eval()
    val_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            predictions = model(xb)
            loss = criterion(predictions, yb)

            val_sum += loss.item() * len(xb)
            val_count += len(xb)

    val_loss = val_sum / val_count

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
        f"Train MSE: {train_loss:.8f} | "
        f"Val MSE: {val_loss:.8f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "window_size": X_train.shape[1],
                "features": [
                    "acc_x_mps2",
                    "acc_y_mps2",
                    "acc_z_mps2",
                    "gyro_x_radps",
                    "gyro_y_radps",
                    "gyro_z_radps",
                ],
                "labels": [
                    "swing_angle_x_rad",
                    "swing_angle_y_rad",
                    "swing_rate_x_radps",
                    "swing_rate_y_radps",
                ],
            },
            checkpoint_path,
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch: {best_epoch}"
        )
        break


# ============================================================
# Plot training and validation loss
# ============================================================

loss_plot_path = LOSS_DIR / "training_loss.png"

plt.figure(figsize=(9, 5))
plt.plot(train_losses, label="Training MSE")
plt.plot(val_losses, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE (rad^2)")
plt.title("LSTM Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(loss_plot_path, dpi=150)
plt.close()


# ============================================================
# Load best model and evaluate on test set
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

predictions_all = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(DEVICE)
        predictions = model(xb)
        predictions_all.append(predictions.cpu().numpy())

y_pred = np.concatenate(predictions_all, axis=0)


def calculate_metrics(y_true, y_hat):
    errors = y_hat - y_true

    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    return mae, rmse


mae, rmse = calculate_metrics(y_test, y_pred)

overall_mae = np.mean(np.abs(y_pred - y_test))
overall_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"Best epoch: {best_epoch}")
print(f"Best validation MSE: {best_val_loss:.8f}")

print(f"X-axis angle MAE:  {mae[0]:.6f} rad ({np.degrees(mae[0]):.3f} deg)")
print(f"Y-axis angle MAE:  {mae[1]:.6f} rad ({np.degrees(mae[1]):.3f} deg)")
print(f"X-axis angle RMSE: {rmse[0]:.6f} rad ({np.degrees(rmse[0]):.3f} deg)")
print(f"Y-axis angle RMSE: {rmse[1]:.6f} rad ({np.degrees(rmse[1]):.3f} deg)")

print(f"X-axis angular velocity MAE: {mae[2]:.6f} rad/s")
print(f"Y-axis angular velocity MAE: {mae[3]:.6f} rsd/s")
print(f"X-axis angular velocity RMSE: {rmse[2]:.6f} rad/s")
print(f"Y-axis angular velocity RMSE: {rmse[3]:.6f} rad/s")

print(f"Overall MAE:  {overall_mae:.6f} rad")
print(f"Overall RMSE: {overall_rmse:.6f} rad")


# ============================================================
# Save predictions and metrics
# ============================================================

np.savez_compressed(
    PREDICTION_DIR / "test_predictions.npz",
    y_true=y_test,
    y_pred=y_pred,
)

metrics_path = METRICS_DIR / "test_metrics.txt"

with metrics_path.open("w", encoding="utf-8") as f:
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Best validation MSE: {best_val_loss:.8f}\n")
    f.write(f"X-axis MAE: {mae[0]:.8f} rad\n")
    f.write(f"Y-axis MAE: {mae[1]:.8f} rad\n")
    f.write(f"X-axis RMSE: {rmse[0]:.8f} rad\n")
    f.write(f"Y-axis RMSE: {rmse[1]:.8f} rad\n")
    f.write(f"Overall MAE: {overall_mae:.8f} rad\n")
    f.write(f"Overall RMSE: {overall_rmse:.8f} rad\n")


# ============================================================
# Plot test predictions
# ============================================================

prediction_plot_path = PREDICTION_DIR / "test_predictions.png"

fig, axes = plt.subplots(2, 2, figsize=(30, 15), sharex=True)

labels = ["Swing angle X", "Swing angle Y", "Swing angular velocity X", "Swing angular velocity Y",]

units = ["Angle (rad)", "Angle (rad)", "Angular velocity (rad/s)", "Angular velocity (rad/s)",]

for i, ax in enumerate(axes.flat):
    ax.plot(
        y_test[:, i],
        label="Ground truth",
        linewidth=1.2,
    )
    ax.plot(
        y_pred[:, i],
        label="LSTM prediction",
        linewidth=1.0,
        alpha=0.85,
    )

    ax.set_ylabel(units[i])
    ax.set_title(labels[i])
    ax.legend()
    ax.grid(True, alpha=0.3)

for ax in axes[1, :]:
    ax.set_xlabel("Test sample index")

plt.tight_layout()
plt.savefig(prediction_plot_path, dpi=150)
plt.close()


print("\nSaved files:")
print(f"Model checkpoint: {checkpoint_path}")
print(f"Training loss:    {loss_plot_path}")
print(f"Test predictions: {prediction_plot_path}")
print(f"Test metrics:     {metrics_path}")
print("\nTraining and evaluation completed.")
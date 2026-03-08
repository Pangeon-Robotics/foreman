"""Train body-state predictors at 3 horizons: 0.1s, 0.2s, 0.5s.

Loads sensor data from collect_training_data.py, trains 3 MLP models that
predict body state residuals (delta from current). Loss = precision-weighted
MSE (surprise).

Input (21D):  body_state(6) + gait_cmd(5) + foot_contacts(4) + imu(6)
Output (6D):  delta body_state at t+horizon

Usage:
    python foreman/train_predictor.py
    python foreman/train_predictor.py --data path/to/body_state_data.npz
    python foreman/train_predictor.py --epochs 500 --batch 512
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BodyStatePredictor(nn.Module):
    """MLP predicting body state residual at a fixed horizon.

    Predicts delta = state(t+h) - state(t) given current observations.
    Also learns per-dimension precision weights (log-precision) so the loss
    naturally emphasizes dimensions that matter most (roll/pitch > vx for
    stability).
    """

    def __init__(self, input_dim=21, output_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
        # Learnable log-precision per output dimension
        # Initialized to equal weighting (log(1)=0)
        self.log_precision = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return self.net(x)

    def precision(self):
        """Precision weights (positive) for weighted MSE."""
        return torch.exp(self.log_precision)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HorizonDataset(Dataset):
    """Pairs (input_t, delta_state_{t+h}) from recorded trajectory data."""

    def __init__(self, body_state, gait_cmd, foot_contacts, imu, horizon_ticks):
        n = len(body_state) - horizon_ticks
        assert n > 0, f"Not enough data for horizon {horizon_ticks} ticks"

        # Input: concat all features at time t
        inputs = np.concatenate([
            body_state[:n],      # 6D
            gait_cmd[:n],        # 5D
            foot_contacts[:n],   # 4D
            imu[:n],             # 6D
        ], axis=1)  # (n, 21)

        # Target: residual body state
        targets = body_state[horizon_ticks:horizon_ticks+n] - body_state[:n]  # (n, 6)

        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

        # Compute normalization stats
        self.input_mean = self.inputs.mean(dim=0)
        self.input_std = self.inputs.std(dim=0).clamp(min=1e-6)
        self.target_mean = self.targets.mean(dim=0)
        self.target_std = self.targets.std(dim=0).clamp(min=1e-6)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = (self.inputs[idx] - self.input_mean) / self.input_std
        y = self.targets[idx]  # don't normalize targets — precision weights handle scaling
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def precision_weighted_mse(pred, target, log_precision):
    """Surprise loss: precision-weighted MSE + precision regularizer.

    L = Σᵢ πᵢ(ŷᵢ - yᵢ)² - log(πᵢ)

    The -log(π) term prevents the model from setting precision to zero
    (ignoring all dimensions). It's equivalent to maximizing the log-likelihood
    of a Gaussian with learned variance.
    """
    precision = torch.exp(log_precision)
    mse_per_dim = (pred - target) ** 2
    weighted = precision * mse_per_dim - log_precision
    return weighted.mean()


def train_one_model(dataset, horizon_name, device, epochs, batch_size, lr, out_dir):
    """Train a single predictor model."""
    # 90/10 train/val split
    n = len(dataset)
    n_val = max(1, n // 10)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = BodyStatePredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = precision_weighted_mse(pred, y, model.log_precision)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = precision_weighted_mse(pred, y, model.log_precision)
                val_loss += loss.item() * len(x)
        val_loss /= n_val

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            prec = model.precision().detach().cpu().numpy()
            prec_str = " ".join(f"{p:.1f}" for p in prec)
            print(f"    [{horizon_name}] epoch {epoch+1:4d}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"prec=[{prec_str}]")

    # Save best model + normalization stats
    model.load_state_dict(best_state)
    save_path = out_dir / f"predictor_{horizon_name}.pt"
    torch.save({
        "model_state": best_state,
        "input_mean": dataset.input_mean,
        "input_std": dataset.input_std,
        "target_mean": dataset.target_mean,
        "target_std": dataset.target_std,
        "horizon_name": horizon_name,
        "best_val_loss": best_val_loss,
    }, save_path)

    return model, best_val_loss


def evaluate_model(model, dataset, horizon_name, device):
    """Print per-dimension prediction error stats."""
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    all_pred = []
    all_target = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu()
            all_pred.append(pred)
            all_target.append(y)

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    errors = (pred - target).numpy()

    dim_names = ["Δvx", "Δvy", "Δwz", "Δroll", "Δpitch", "Δyaw"]
    prec = model.precision().detach().cpu().numpy()

    print(f"\n  [{horizon_name}] Per-dimension errors (RMSE / MAE / learned precision):")
    for i, name in enumerate(dim_names):
        rmse = np.sqrt(np.mean(errors[:, i] ** 2))
        mae = np.mean(np.abs(errors[:, i]))
        print(f"    {name:>7s}: RMSE={rmse:.6f}  MAE={mae:.6f}  π={prec[i]:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    _root = Path(__file__).resolve().parents[1]
    data_path = args.data or str(_root / "foreman" / "tmp" / "training_data" / "body_state_data.npz")
    out_dir = _root / "foreman" / "tmp" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        print("Run: python foreman/collect_training_data.py")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    body_state = data["body_state"]
    gait_cmd = data["gait_cmd"]
    foot_contacts = data["foot_contacts"]
    imu = data["imu"]
    dt = float(data["dt"])

    print(f"  {len(body_state)} samples at dt={dt}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Three horizons
    horizons = {
        "100ms": int(0.1 / dt),   # 10 ticks
        "200ms": int(0.2 / dt),   # 20 ticks
        "500ms": int(0.5 / dt),   # 50 ticks
    }

    print(f"\n{'='*70}")
    print(f"Training 3 predictors: {list(horizons.keys())}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    results = {}
    for name, ticks in horizons.items():
        if ticks >= len(body_state):
            print(f"\n  [{name}] SKIP — need {ticks} ticks but only have {len(body_state)}")
            continue

        print(f"\n  [{name}] Building dataset (horizon={ticks} ticks)...")
        ds = HorizonDataset(body_state, gait_cmd, foot_contacts, imu, ticks)
        print(f"    {len(ds)} training pairs")

        model, val_loss = train_one_model(
            ds, name, device, args.epochs, args.batch, args.lr, out_dir)
        evaluate_model(model, ds, name, device)
        results[name] = val_loss

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, loss in results.items():
        path = out_dir / f"predictor_{name}.pt"
        size_kb = path.stat().st_size / 1024
        print(f"  {name:>5s}: val_loss={loss:.6f}  saved={path.name} ({size_kb:.0f}KB)")

    print(f"\nModels saved to {out_dir}/")
    print("Next: plug into layer_5/generative_model.py as drop-in predictor")


if __name__ == "__main__":
    main()

"""Physics-informed neural network scaffold for stellar-profile inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - torch is an optional heavy dependency at import time.
    torch = None
    nn = None
    DataLoader = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for stellar_analyzer.ml.pinn_model")


def build_input_features(mass, teff, metallicity, age, radius_fraction):
    """Build the 15 PINN input features requested by the prompt."""

    values = [
        np.asarray(mass, dtype=float),
        np.asarray(teff, dtype=float) / 10000.0,
        np.asarray(metallicity, dtype=float),
        np.asarray(age, dtype=float) / 10.0,
        np.asarray(radius_fraction, dtype=float),
    ]
    r = values[-1]
    m = values[0]
    t = values[1]
    z = values[2]
    a = values[3]
    features = values + [
        r**2,
        np.sqrt(np.clip(r, 0.0, None)),
        np.sin(np.pi * r),
        np.cos(np.pi * r),
        np.sin(2.0 * np.pi * r),
        np.cos(2.0 * np.pi * r),
        np.log1p(np.clip(m, 0.0, None)),
        t**4,
        z * r,
        a * r,
    ]
    return np.stack(features, axis=-1)


class RadialAttention(nn.Module if nn is not None else object):
    """A compact attention layer that conditions physical features on radius."""

    def __init__(self, width: int):
        _require_torch()
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.scale = width**-0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = torch.softmax((q * k).sum(dim=-1, keepdim=True) * self.scale, dim=1 if x.ndim == 3 else -1)
        return x + weights * v


class StellarPINN(nn.Module if nn is not None else object):
    """Six-layer Tanh PINN with decoupled deviation and radial-profile heads."""

    def __init__(self, input_dim: int = 15, width: int = 256, depth: int = 6):
        _require_torch()
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, width), nn.Tanh()])
            current = width
        self.trunk = nn.Sequential(*layers)
        self.attention = RadialAttention(width)
        self.delta_head = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 5))
        self.profile_head = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 3))

    def forward(self, x):
        hidden = self.trunk(x)
        hidden = self.attention(hidden)
        profile = self.profile_head(hidden)
        if hidden.ndim == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden
        delta = self.delta_head(pooled)
        return {"delta_n": delta, "profiles": profile}


def physics_residual_loss(radius_fraction, theta, n_index):
    """Lane-Emden residual loss computed with autograd."""

    _require_torch()
    grad_theta = torch.autograd.grad(theta.sum(), radius_fraction, create_graph=True, retain_graph=True)[0]
    flux = radius_fraction**2 * grad_theta
    grad_flux = torch.autograd.grad(flux.sum(), radius_fraction, create_graph=True, retain_graph=True)[0]
    residual = grad_flux / torch.clamp(radius_fraction**2, min=1e-6) + torch.clamp(theta, min=0.0) ** n_index
    return torch.mean(residual**2)


@dataclass
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    lambda_physics: float = 1e-2
    patience: int = 20
    output_path: str = "models/pinn_weights.pt"


def train_pinn(model: StellarPINN, train_dataset, val_dataset=None, config: TrainingConfig | None = None) -> dict:
    """Train the PINN on preprocessed stellar-grid tensors.

    Dataset batches should provide ``features``, ``delta_n``, ``profiles``, and
    optionally ``radius_fraction`` and ``theta`` for the physics residual.
    """

    _require_torch()
    config = config or TrainingConfig()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size) if val_dataset is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    mse = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            features = batch["features"].float()
            target_delta = batch["delta_n"].float()
            target_profiles = batch["profiles"].float()
            optimizer.zero_grad()
            pred = model(features)
            loss = mse(pred["delta_n"], target_delta) + mse(pred["profiles"], target_profiles)
            if "radius_fraction" in batch and "theta" in batch:
                radius_fraction = batch["radius_fraction"].float().requires_grad_(True)
                theta = batch["theta"].float()
                loss = loss + config.lambda_physics * physics_residual_loss(radius_fraction, theta, n_index=1.5)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(train_losses))
        if val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    pred = model(batch["features"].float())
                    losses.append(float(mse(pred["delta_n"], batch["delta_n"].float()) + mse(pred["profiles"], batch["profiles"].float())))
            val_loss = float(np.mean(losses))
        scheduler.step(val_loss)
        history.append({"epoch": float(epoch), "loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return {"best_loss": best_loss, "epochs_ran": len(history), "history": history, "weights": str(output_path)}


def load_pinn_weights(path: str | Path = "models/pinn_weights.pt", map_location: str = "cpu") -> StellarPINN:
    _require_torch()
    model = StellarPINN()
    weights_path = Path(path)
    if weights_path.exists() and weights_path.stat().st_size > 0:
        model.load_state_dict(torch.load(weights_path, map_location=map_location))
    model.eval()
    return model

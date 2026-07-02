"""Physics-informed sequence model and reproducible training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    Subset = None


DELTA_NAMES = ("delta_n_conv", "delta_n_deg", "delta_n_mu", "delta_n_nuc", "delta_n_rad")


def transform_delta_n(values):
    """Compress wide dynamic ranges while preserving sign and zero."""

    if torch is not None and isinstance(values, torch.Tensor):
        return torch.sign(values) * torch.log1p(torch.abs(values))
    values = np.asarray(values)
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_transform_delta_n(values):
    """Restore signed-log deviation targets to their physical values."""

    if torch is not None and isinstance(values, torch.Tensor):
        return torch.sign(values) * torch.expm1(torch.abs(values))
    values = np.asarray(values)
    return np.sign(values) * np.expm1(np.abs(values))


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required; install the training dependencies first")


def build_input_features(mass, teff, metallicity, age, radius_fraction):
    """Build the 15 dimensionless features used by the PINN."""

    values = [
        np.asarray(mass, dtype=float),
        np.asarray(teff, dtype=float) / 10000.0,
        np.asarray(metallicity, dtype=float),
        np.asarray(age, dtype=float) / 10.0,
        np.asarray(radius_fraction, dtype=float),
    ]
    m, t, z, a, r = values
    smooth_root = (np.sqrt(r + 0.01) - 0.1) / (np.sqrt(1.01) - 0.1)
    return np.stack(
        values
        + [r**2, smooth_root, np.sin(np.pi * r), np.cos(np.pi * r),
           np.sin(2.0 * np.pi * r), np.cos(2.0 * np.pi * r), np.log1p(np.clip(m, 0.0, None)),
           t**4, z * r, a * r],
        axis=-1,
    )


def build_differentiable_features(stored_features):
    """Reconnect engineered radial columns to a differentiable radius tensor."""

    m, t, z, a = (stored_features[..., index].detach() for index in range(4))
    radius = stored_features[..., 4].detach().requires_grad_(True)
    smooth_root = (torch.sqrt(radius + 0.01) - 0.1) / (np.sqrt(1.01) - 0.1)
    features = torch.stack(
        [m, t, z, a, radius, radius**2, smooth_root,
         torch.sin(torch.pi * radius), torch.cos(torch.pi * radius),
         torch.sin(2.0 * torch.pi * radius), torch.cos(2.0 * torch.pi * radius),
         torch.log1p(torch.clamp(m, min=0.0)), t**4, z * radius, a * radius],
        dim=-1,
    )
    return features, radius


class RadialAttention(nn.Module if nn is not None else object):
    def __init__(self, width: int):
        _require_torch()
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))
        self.value = nn.Linear(width, width)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        context = torch.sum(weights * self.value(x), dim=1, keepdim=True)
        return x + context


class StellarPINN(nn.Module if nn is not None else object):
    """Radial sequence PINN with global deviation and profile heads."""

    def __init__(self, input_dim: int = 15, width: int = 128, depth: int = 5):
        _require_torch()
        super().__init__()
        self.architecture = {"input_dim": input_dim, "width": width, "depth": depth}
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
        if x.ndim != 3:
            raise ValueError("PINN input must have shape [batch, radial_points, features]")
        hidden = self.trunk(x)
        attended = self.attention(hidden)
        return {"delta_n": self.delta_head(attended.mean(dim=1)), "profiles": self.profile_head(hidden)}


class NpzStellarDataset(Dataset):
    """In-memory, non-pickled dataset produced by ``prepare-pinn``."""

    def __init__(self, path: str | Path):
        _require_torch()
        with np.load(path, allow_pickle=False) as data:
            self.features = torch.from_numpy(data["features"].astype(np.float32))
            self.profiles = torch.from_numpy(data["profiles"].astype(np.float32))
            self.delta_n = torch.from_numpy(data["delta_n"].astype(np.float32))
            self.n_index = torch.from_numpy(data["n_index"].astype(np.float32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {"features": self.features[index], "profiles": self.profiles[index],
                "delta_n": self.delta_n[index], "n_index": self.n_index[index]}


def physics_residual_loss(radius, theta, n_index):
    """Lane-Emden residual using model-predicted theta and the radial feature."""

    grad = torch.autograd.grad(theta.sum(), radius, create_graph=True, retain_graph=True)[0]
    second = torch.autograd.grad(grad.sum(), radius, create_graph=True, retain_graph=True)[0]
    residual = second + 2.0 * grad / torch.clamp(radius, min=0.03) + torch.clamp(theta, min=0.0) ** n_index[:, None]
    interior = radius > 0.03
    equation_loss = torch.mean(residual[interior] ** 2)
    boundary_loss = torch.mean((theta[:, 0] - 1.0) ** 2 + grad[:, 0] ** 2 + theta[:, -1] ** 2)
    return equation_loss + boundary_loss


@dataclass
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 4
    learning_rate: float = 1e-3
    lambda_physics: float = 1e-2
    patience: int = 30
    seed: int = 42
    device: str = "auto"
    output_path: str = "models/pinn_checkpoint.pt"


def _split_dataset(dataset, seed: int):
    if len(dataset) < 3:
        raise ValueError("Training requires at least three samples")
    indices = np.random.default_rng(seed).permutation(len(dataset)).tolist()
    val_count = max(1, round(len(indices) * 0.15))
    test_count = max(1, round(len(indices) * 0.15))
    train_count = len(indices) - val_count - test_count
    if train_count < 1:
        raise ValueError("Dataset is too small for train/validation/test splits")
    return (Subset(dataset, indices[:train_count]),
            Subset(dataset, indices[train_count:train_count + val_count]),
            Subset(dataset, indices[train_count + val_count:]))


def _supervised_loss(prediction, batch, mse):
    return mse(prediction["delta_n"], batch["delta_n"]) + mse(prediction["profiles"], batch["profiles"])


def _evaluate(model, loader, device, mse):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(_supervised_loss(model(batch["features"]), batch, mse).cpu()))
    return float(np.mean(losses))


def train_pinn(dataset, config: TrainingConfig | None = None, model: StellarPINN | None = None) -> dict:
    """Train, validate, test, and save a metadata-rich checkpoint."""

    _require_torch()
    config = config or TrainingConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                          "cpu" if config.device == "auto" else config.device)
    model = (model or StellarPINN()).to(device)
    train_set, val_set, test_set = _split_dataset(dataset, config.seed)
    train_loader = DataLoader(train_set, batch_size=min(config.batch_size, len(train_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    test_loader = DataLoader(test_set, batch_size=len(test_set))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    mse = nn.MSELoss()
    best_loss, best_state, stale_epochs, history = float("inf"), None, 0, []

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            features, radius = build_differentiable_features(batch["features"])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            supervised = _supervised_loss(prediction, batch, mse)
            physics = physics_residual_loss(radius, prediction["profiles"][..., 0], batch["n_index"])
            loss = supervised + config.lambda_physics * physics
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation = _evaluate(model, val_loader, device, mse)
        scheduler.step(validation)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": validation})
        if validation < best_loss:
            best_loss = validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    test_loss = _evaluate(model, test_loader, device, mse)
    metrics = {"best_validation_loss": best_loss, "test_loss": test_loss, "epochs_ran": len(history),
               "split_sizes": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)}}
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "state_dict": best_state, "architecture": model.architecture,
                "training_config": asdict(config), "metrics": metrics, "history": history}, output)
    return {**metrics, "weights": str(output), "device": str(device)}


def load_pinn_weights(path: str | Path = "models/pinn_checkpoint.pt", map_location: str = "cpu") -> StellarPINN:
    _require_torch()
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "state_dict" in checkpoint:
        model = StellarPINN(**checkpoint.get("architecture", {}))
        model.load_state_dict(checkpoint["state_dict"])
    else:  # Compatibility with early state-dict-only checkpoints.
        model = StellarPINN()
        model.load_state_dict(checkpoint)
    model.eval()
    return model

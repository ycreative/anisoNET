"""Physics-informed neural network solvers for anisoNET."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PINNProfile:
    """Training profile for different hardware budgets."""

    name: str
    hidden_width: int
    hidden_depth: int
    num_domain: int
    num_boundary: int
    iterations: int
    learning_rate: float
    data_weight: float
    boundary_weight: float
    pde_weight: float
    background_weight: float
    smoothness_weight: float
    display_every: int
    network: str = "mlp"
    fourier_features: int = 0
    fourier_sigma: float = 10.0


PINN_PROFILES = {
    "debug_fit": PINNProfile(
        name="debug_fit",
        hidden_width=128,
        hidden_depth=3,
        num_domain=1500,
        num_boundary=80,
        iterations=1000,
        learning_rate=1e-3,
        data_weight=5.0,
        boundary_weight=0.01,
        pde_weight=0.05,
        background_weight=1.0,
        smoothness_weight=0.0,
        display_every=200,
    ),
    "debug_fourier": PINNProfile(
        name="debug_fourier",
        hidden_width=128,
        hidden_depth=3,
        num_domain=2000,
        num_boundary=80,
        iterations=1500,
        learning_rate=1e-3,
        data_weight=8.0,
        boundary_weight=0.01,
        pde_weight=0.02,
        background_weight=1.0,
        smoothness_weight=0.0,
        display_every=300,
        network="fourier",
        fourier_features=64,
        fourier_sigma=8.0,
    ),
    "fourier_balanced": PINNProfile(
        name="fourier_balanced",
        hidden_width=128,
        hidden_depth=3,
        num_domain=3000,
        num_boundary=100,
        iterations=2000,
        learning_rate=8e-4,
        data_weight=6.0,
        boundary_weight=0.02,
        pde_weight=0.08,
        background_weight=1.5,
        smoothness_weight=0.002,
        display_every=400,
        network="fourier",
        fourier_features=48,
        fourier_sigma=4.0,
    ),
    "fourier_refined_16g": PINNProfile(
        name="fourier_refined_16g",
        hidden_width=128,
        hidden_depth=3,
        num_domain=2000,
        num_boundary=100,
        iterations=1000,
        learning_rate=8e-4,
        data_weight=8.0,
        boundary_weight=0.02,
        pde_weight=0.12,
        background_weight=1.5,
        smoothness_weight=0.002,
        display_every=500,
        network="fourier",
        fourier_features=48,
        fourier_sigma=6.5,
    ),
    "fourier_refined_low_pde_16g": PINNProfile(
        name="fourier_refined_low_pde_16g",
        hidden_width=128,
        hidden_depth=3,
        num_domain=2000,
        num_boundary=100,
        iterations=1000,
        learning_rate=8e-4,
        data_weight=8.0,
        boundary_weight=0.02,
        pde_weight=0.04,
        background_weight=1.5,
        smoothness_weight=0.002,
        display_every=500,
        network="fourier",
        fourier_features=48,
        fourier_sigma=6.5,
    ),
    "smoke": PINNProfile(
        name="smoke",
        hidden_width=64,
        hidden_depth=2,
        num_domain=1000,
        num_boundary=80,
        iterations=200,
        learning_rate=1e-3,
        data_weight=0.05,
        boundary_weight=0.1,
        pde_weight=1.0,
        background_weight=0.1,
        smoothness_weight=0.0,
        display_every=100,
    ),
    "local_16g": PINNProfile(
        name="local_16g",
        hidden_width=128,
        hidden_depth=3,
        num_domain=8000,
        num_boundary=200,
        iterations=2500,
        learning_rate=1e-3,
        data_weight=0.05,
        boundary_weight=0.1,
        pde_weight=1.0,
        background_weight=0.1,
        smoothness_weight=0.0,
        display_every=500,
    ),
    "l20_24g": PINNProfile(
        name="l20_24g",
        hidden_width=256,
        hidden_depth=4,
        num_domain=25000,
        num_boundary=400,
        iterations=8000,
        learning_rate=1e-3,
        data_weight=0.05,
        boundary_weight=0.1,
        pde_weight=1.0,
        background_weight=0.1,
        smoothness_weight=0.0,
        display_every=1000,
    ),
}


@dataclass(frozen=True)
class PINNResult:
    spot_prediction: np.ndarray
    grid_prediction: np.ndarray
    history: dict
    profile: PINNProfile


def get_profile(name: str) -> PINNProfile:
    """Return a named PINN training profile."""

    try:
        return PINN_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PINN_PROFILES))
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}") from exc


def robust_normalize(values: np.ndarray, *, percentile: float = 99.5) -> np.ndarray:
    """Robustly normalize a vector or grid to [0, 1]."""

    arr = np.asarray(values, dtype=np.float32)
    vmax = np.percentile(arr, percentile)
    vmin = np.min(arr)
    return (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)


def solve_scalar_reaction_diffusion(
    coords_norm: np.ndarray,
    source_values: np.ndarray,
    scalar_diffusion_grid: np.ndarray,
    source_grid: np.ndarray,
    *,
    profile: PINNProfile,
    tissue_mask: Optional[np.ndarray] = None,
    prior_grid: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
    prior_mode: str = "loss",
    k: float = 2.5,
    device: Optional[str] = None,
    seed: int = 0,
    prediction_grid_size: int = 200,
) -> PINNResult:
    """Solve a scalar reaction-diffusion PDE with native PyTorch.

    The implemented PDE follows the historical anisoNET/SASP-Agent prototype:

    ``D(x, y) * (C_xx + C_yy) - k * C + S(x, y) = 0``.

    This function intentionally names the input ``scalar_diffusion_grid`` to
    avoid implying a full tensor formulation.
    """

    import torch

    if prior_mode not in {"loss", "residual_anchor"}:
        raise ValueError("prior_mode must be 'loss' or 'residual_anchor'")
    if prior_mode == "residual_anchor" and prior_grid is None:
        raise ValueError("prior_grid is required when prior_mode='residual_anchor'")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    coords = np.asarray(coords_norm, dtype=np.float32)
    values = np.asarray(source_values, dtype=np.float32).reshape(-1, 1)
    grid_size = int(scalar_diffusion_grid.shape[0])
    if scalar_diffusion_grid.shape != source_grid.shape:
        raise ValueError("scalar_diffusion_grid and source_grid must have matching shapes")
    if prior_grid is not None and scalar_diffusion_grid.shape != prior_grid.shape:
        raise ValueError("prior_grid must match scalar_diffusion_grid shape")
    if scalar_diffusion_grid.shape[0] != scalar_diffusion_grid.shape[1]:
        raise ValueError("Only square grids are currently supported")

    diffusion_tensor = torch.tensor(scalar_diffusion_grid, dtype=torch.float32, device=torch_device)
    source_tensor = torch.tensor(source_grid, dtype=torch.float32, device=torch_device)
    prior_tensor = (
        torch.tensor(prior_grid, dtype=torch.float32, device=torch_device)
        if prior_grid is not None and prior_weight > 0
        else None
    )
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=torch_device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=torch_device)

    model = _build_network(
        hidden_width=profile.hidden_width,
        hidden_depth=profile.hidden_depth,
        network=profile.network,
        fourier_features=profile.fourier_features,
        fourier_sigma=profile.fourier_sigma,
        output_activation="linear" if prior_mode == "residual_anchor" else "softplus",
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate)

    domain_points = torch.rand(profile.num_domain, 2, dtype=torch.float32, device=torch_device)
    boundary_points = _sample_boundary(profile.num_boundary, torch_device)

    if tissue_mask is not None:
        tissue_mask_bool = np.asarray(tissue_mask, dtype=bool)
        domain_np = _sample_mask_points(tissue_mask_bool, profile.num_domain, seed=seed)
        background_np = _sample_mask_points(~tissue_mask_bool, max(profile.num_domain // 4, 1), seed=seed + 17)
        domain_points = torch.tensor(domain_np, dtype=torch.float32, device=torch_device)
        background_points = torch.tensor(background_np, dtype=torch.float32, device=torch_device)
    else:
        background_points = torch.empty(0, 2, dtype=torch.float32, device=torch_device)

    history = {
        "iteration": [],
        "loss": [],
        "loss_pde": [],
        "loss_data": [],
        "loss_boundary": [],
        "loss_background": [],
        "loss_smoothness": [],
        "loss_prior": [],
    }
    for iteration in range(1, profile.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_pde = _pde_loss(
            model,
            domain_points,
            diffusion_tensor,
            source_tensor,
            prior_grid=prior_tensor,
            prior_mode=prior_mode,
            k=float(k),
        )
        loss_smoothness = _smoothness_loss(model, domain_points, prior_grid=prior_tensor, prior_mode=prior_mode)
        pred_data = _model_prediction(model, coords_tensor, prior_grid=prior_tensor, prior_mode=prior_mode)
        loss_data = torch.mean((pred_data - values_tensor) ** 2)
        pred_boundary = _model_prediction(model, boundary_points, prior_grid=prior_tensor, prior_mode=prior_mode)
        loss_boundary = torch.mean(pred_boundary**2)
        if prior_tensor is not None:
            if prior_mode == "residual_anchor":
                loss_prior = torch.mean(model(domain_points) ** 2)
            else:
                prior_values = _grid_lookup(prior_tensor, domain_points)
                pred_prior = _model_prediction(model, domain_points, prior_grid=prior_tensor, prior_mode=prior_mode)
                loss_prior = torch.mean((pred_prior - prior_values) ** 2)
        else:
            loss_prior = torch.tensor(0.0, dtype=torch.float32, device=torch_device)
        if background_points.shape[0] > 0:
            pred_background = _model_prediction(model, background_points, prior_grid=prior_tensor, prior_mode=prior_mode)
            loss_background = torch.mean(pred_background**2)
        else:
            loss_background = torch.tensor(0.0, dtype=torch.float32, device=torch_device)
        loss = (
            profile.pde_weight * loss_pde
            + profile.data_weight * loss_data
            + profile.boundary_weight * loss_boundary
            + profile.background_weight * loss_background
            + profile.smoothness_weight * loss_smoothness
            + float(prior_weight) * loss_prior
        )
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % profile.display_every == 0 or iteration == profile.iterations:
            history["iteration"].append(iteration)
            history["loss"].append(float(loss.detach().cpu()))
            history["loss_pde"].append(float(loss_pde.detach().cpu()))
            history["loss_data"].append(float(loss_data.detach().cpu()))
            history["loss_boundary"].append(float(loss_boundary.detach().cpu()))
            history["loss_background"].append(float(loss_background.detach().cpu()))
            history["loss_smoothness"].append(float(loss_smoothness.detach().cpu()))
            history["loss_prior"].append(float(loss_prior.detach().cpu()))
            print(
                f"[{profile.name}] iter={iteration} "
                f"loss={history['loss'][-1]:.6g} "
                f"pde={history['loss_pde'][-1]:.6g} "
                f"data={history['loss_data'][-1]:.6g} "
                f"bc={history['loss_boundary'][-1]:.6g} "
                f"bg={history['loss_background'][-1]:.6g} "
                f"smooth={history['loss_smoothness'][-1]:.6g} "
                f"prior={history['loss_prior'][-1]:.6g}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        spot_prediction = (
            _model_prediction(model, coords_tensor, prior_grid=prior_tensor, prior_mode=prior_mode)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    grid_axis = np.linspace(0, 1, prediction_grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid_coords = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_coords, dtype=torch.float32, device=torch_device)
        grid_prediction = (
            _model_prediction(model, grid_tensor, prior_grid=prior_tensor, prior_mode=prior_mode)
            .detach()
            .cpu()
            .numpy()
            .reshape(prediction_grid_size, prediction_grid_size)
        )

    return PINNResult(
        spot_prediction=spot_prediction.astype(np.float32),
        grid_prediction=grid_prediction.astype(np.float32),
        history=history,
        profile=profile,
    )


def _build_network(
    *,
    hidden_width: int,
    hidden_depth: int,
    network: str = "mlp",
    fourier_features: int = 0,
    fourier_sigma: float = 10.0,
    output_activation: str = "softplus",
):
    import torch

    if network == "fourier":
        return FourierFeatureNetwork(
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
            fourier_features=fourier_features,
            sigma=fourier_sigma,
            output_activation=output_activation,
        )
    if network != "mlp":
        raise ValueError("network must be either 'mlp' or 'fourier'")

    layers = []
    in_features = 2
    for _ in range(hidden_depth):
        linear = torch.nn.Linear(in_features, hidden_width)
        torch.nn.init.xavier_uniform_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        layers.extend([linear, torch.nn.Tanh()])
        in_features = hidden_width
    output = torch.nn.Linear(in_features, 1)
    torch.nn.init.xavier_uniform_(output.weight)
    torch.nn.init.zeros_(output.bias)
    layers.append(output)
    if output_activation == "softplus":
        layers.append(torch.nn.Softplus())
    elif output_activation != "linear":
        raise ValueError("output_activation must be 'softplus' or 'linear'")
    return torch.nn.Sequential(*layers)


class FourierFeatureNetwork:
    """MLP with fixed random Fourier features."""

    def __init__(
        self,
        *,
        hidden_width: int,
        hidden_depth: int,
        fourier_features: int,
        sigma: float,
        output_activation: str = "softplus",
    ):
        import torch

        super().__init__()
        self.module = None
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.fourier_features = int(fourier_features)
        self.sigma = float(sigma)
        if self.fourier_features <= 0:
            raise ValueError("fourier_features must be positive for FourierFeatureNetwork")
        self._torch = torch
        self.B = torch.randn(2, self.fourier_features) * self.sigma
        layers = []
        in_features = self.fourier_features * 2
        for _ in range(hidden_depth):
            linear = torch.nn.Linear(in_features, hidden_width)
            torch.nn.init.xavier_uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            layers.extend([linear, torch.nn.Tanh()])
            in_features = hidden_width
        output = torch.nn.Linear(in_features, 1)
        torch.nn.init.xavier_uniform_(output.weight)
        torch.nn.init.zeros_(output.bias)
        layers.append(output)
        if output_activation == "softplus":
            layers.append(torch.nn.Softplus())
        elif output_activation != "linear":
            raise ValueError("output_activation must be 'softplus' or 'linear'")
        self.module = torch.nn.Sequential(*layers)

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        self.B = self.B.to(*args, **kwargs)
        return self

    def parameters(self):
        return self.module.parameters()

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def __call__(self, x):
        import torch

        projected = 2.0 * np.pi * x @ self.B
        encoded = torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)
        return self.module(encoded)


def _sample_boundary(num_boundary: int, device):
    import torch

    if num_boundary <= 0:
        return torch.empty(0, 2, dtype=torch.float32, device=device)
    n_each = max(1, num_boundary // 4)
    t = torch.rand(n_each, 1, dtype=torch.float32, device=device)
    left = torch.cat([torch.zeros_like(t), t], dim=1)
    right = torch.cat([torch.ones_like(t), t], dim=1)
    bottom = torch.cat([t, torch.zeros_like(t)], dim=1)
    top = torch.cat([t, torch.ones_like(t)], dim=1)
    points = torch.cat([left, right, bottom, top], dim=0)
    if points.shape[0] > num_boundary:
        points = points[:num_boundary]
    return points


def _sample_mask_points(mask: np.ndarray, count: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return rng.random((count, 2), dtype=np.float32)
    selected = rng.integers(0, len(x_idx), size=count)
    jitter = rng.random((count, 2), dtype=np.float32)
    grid_size = mask.shape[0]
    x = (x_idx[selected].astype(np.float32) + jitter[:, 0]) / max(grid_size - 1, 1)
    y = (y_idx[selected].astype(np.float32) + jitter[:, 1]) / max(grid_size - 1, 1)
    return np.column_stack([np.clip(x, 0, 1), np.clip(y, 0, 1)]).astype(np.float32)


def _grid_lookup(grid, points):
    import torch

    scale = grid.shape[0] - 1
    x_idx = torch.clamp((points[:, 0] * scale).long(), 0, scale)
    y_idx = torch.clamp((points[:, 1] * scale).long(), 0, scale)
    return grid[y_idx, x_idx].unsqueeze(1)


def _model_prediction(model, points, *, prior_grid=None, prior_mode: str = "loss"):
    if prior_grid is not None and prior_mode == "residual_anchor":
        return _grid_lookup(prior_grid, points) + model(points)
    return model(points)


def _pde_loss(model, points, diffusion_grid, source_grid, *, k: float, prior_grid=None, prior_mode: str = "loss"):
    import torch

    x = points.detach().clone().requires_grad_(True)
    c = _model_prediction(model, x, prior_grid=prior_grid, prior_mode=prior_mode)
    grad_c = torch.autograd.grad(
        c,
        x,
        grad_outputs=torch.ones_like(c),
        create_graph=True,
        retain_graph=True,
    )[0]
    c_x = grad_c[:, 0:1]
    c_y = grad_c[:, 1:2]
    c_xx = torch.autograd.grad(
        c_x,
        x,
        grad_outputs=torch.ones_like(c_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    c_yy = torch.autograd.grad(
        c_y,
        x,
        grad_outputs=torch.ones_like(c_y),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    d_val = _grid_lookup(diffusion_grid, x)
    s_val = _grid_lookup(source_grid, x)
    residual = d_val * (c_xx + c_yy) - k * c + s_val
    return torch.mean(residual**2)


def _smoothness_loss(model, points, *, prior_grid=None, prior_mode: str = "loss"):
    import torch

    x = points.detach().clone().requires_grad_(True)
    c = _model_prediction(model, x, prior_grid=prior_grid, prior_mode=prior_mode)
    grad_c = torch.autograd.grad(
        c,
        x,
        grad_outputs=torch.ones_like(c),
        create_graph=True,
        retain_graph=True,
    )[0]
    return torch.mean(torch.sum(grad_c**2, dim=1, keepdim=True))

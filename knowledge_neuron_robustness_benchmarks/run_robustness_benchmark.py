"""Parameter-efficiency and robustness benchmarks for knowledge-neuron models.

This experiment package studies two settings where domain knowledge can be more
valuable than a small improvement at one SNR point:

1. DetNet-style MIMO detection under channel/noise distribution shifts.
2. DeepRx-inspired OFDM reception under delay-spread, CFO, and impulsive-noise
   shifts.

Each scenario compares one parameter-rich paper-inspired NN against several
smaller knowledge-neuron networks. The report includes task metrics, parameter
counts, training time, sample-efficiency sweeps, and robustness gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


SEED = 23
MIMO_N_TX = 8
MIMO_N_RX = 8
OFDM_T = 8
OFDM_K = 64
OFDM_PILOT_SPACING = 4


SCENARIO_META = {
    "mimo": {
        "title": "MIMO detection under distribution shift",
        "metric": "BER",
        "paper": "DetNet-style unfolded detector inspired by Samuel, Diskin, and Wiesel, Deep MIMO Detection (2017).",
        "description": "A real-valued 8x8 BPSK MIMO detector is trained on iid Rayleigh channels and tested on correlated, Rician, impulsive-noise, noise-mismatch, and SNR-shift conditions.",
        "robust_condition": "correlated",
        "conditions": {
            "id": "IID Rayleigh, correctly specified noise",
            "correlated": "Strongly correlated Rayleigh channel",
            "rician": "Rician channel with a line-of-sight component",
            "noise_mismatch": "Noise variance supplied to the receiver is wrong",
            "impulsive": "Rare impulsive noise outliers",
            "low_snr": "IID Rayleigh at 0 dB",
            "high_snr": "IID Rayleigh at 20 dB",
        },
    },
    "ofdm": {
        "title": "OFDM receiver under waveform impairment shift",
        "metric": "BER",
        "paper": "DeepRx-inspired fully convolutional receiver inspired by Honkala, Korpi, and Huttunen, DeepRx (2020).",
        "description": "A 64-subcarrier QPSK OFDM receiver is trained on nominal short-delay channels and tested under longer delay spread, carrier-frequency offset, impulsive noise, and SNR shifts.",
        "robust_condition": "cfo",
        "conditions": {
            "id": "Nominal short-delay channel",
            "long_delay": "Longer delay spread than training",
            "cfo": "Unseen carrier-frequency offset",
            "impulsive": "Rare impulsive time-frequency noise",
            "low_snr": "Nominal channel at 0 dB",
            "high_snr": "Nominal channel at 20 dB",
        },
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mmse_estimate(H: Tensor, y: Tensor, sigma2: Tensor) -> Tensor:
    batch, n_tx = H.shape[0], H.shape[2]
    eye = torch.eye(n_tx, device=H.device, dtype=H.dtype).expand(batch, -1, -1)
    gram = H.transpose(1, 2) @ H
    rhs = (H.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(gram + sigma2[:, None, None] * eye, rhs.unsqueeze(-1)).squeeze(-1)


def make_mimo_data(
    n: int,
    snr_db: float | None,
    condition: str,
    seed: int,
) -> dict[str, Tensor]:
    rng = np.random.default_rng(seed)
    snr = np.full(n, 10.0 if snr_db is None else snr_db, dtype=np.float32)
    if snr_db is None:
        snr = rng.uniform(0.0, 20.0, size=n).astype(np.float32)
    H = rng.standard_normal((n, MIMO_N_RX, MIMO_N_TX)).astype(np.float32) / math.sqrt(MIMO_N_RX)
    if condition == "correlated":
        rho = 0.82
        corr = rho ** np.abs(np.arange(MIMO_N_RX)[:, None] - np.arange(MIMO_N_RX)[None, :])
        H = np.einsum("ij,bjk->bik", np.linalg.cholesky(corr).astype(np.float32), H)
    elif condition == "rician":
        los = np.ones((n, MIMO_N_RX, MIMO_N_TX), dtype=np.float32) / math.sqrt(MIMO_N_RX)
        H = math.sqrt(0.45) * los + math.sqrt(0.55) * H
    x = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, MIMO_N_TX))
    clean = np.einsum("bij,bj->bi", H, x)
    signal_power = np.mean(clean**2, axis=1).astype(np.float32)
    actual_var = signal_power / np.power(10.0, snr / 10.0)
    noise = rng.standard_normal((n, MIMO_N_RX)).astype(np.float32) * np.sqrt(actual_var[:, None])
    if condition == "impulsive":
        outlier = rng.random((n, MIMO_N_RX)) < 0.035
        noise += outlier.astype(np.float32) * rng.standard_normal((n, MIMO_N_RX)).astype(np.float32) * np.sqrt(actual_var[:, None]) * 7.0
    y = clean + noise
    model_var = actual_var.copy()
    if condition == "noise_mismatch":
        model_var *= 0.35
    return {
        "H": torch.from_numpy(H),
        "y": torch.from_numpy(y),
        "x": torch.from_numpy(x),
        "sigma2": torch.from_numpy(model_var),
    }


class PlainDetNetStage(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_tx: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_tx)

    def forward(self, features: Tensor, _prior: Tensor | None = None) -> Tensor:
        z = F.relu(self.fc1(features))
        z = F.relu(self.fc2(z))
        return self.out(z)


class KnowledgeNeuronDetNetStage(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_tx: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.prior_projection = nn.Linear(n_tx, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_tx)
        self.trust_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, features: Tensor, prior: Tensor) -> Tensor:
        z = self.fc1(features)
        knowledge = torch.tanh(self.prior_projection(prior))
        z = F.relu(z) + torch.sigmoid(self.trust_logit) * knowledge
        z = F.relu(self.fc2(z))
        return self.out(z)


class DetNetStyle(nn.Module):
    """DetNet-like unfolded gradient detector with optional knowledge neuron."""

    def __init__(self, depth: int, hidden: int, knowledge: bool):
        super().__init__()
        self.knowledge = knowledge
        input_dim = 4 * MIMO_N_TX if knowledge else 3 * MIMO_N_TX
        stage_type = KnowledgeNeuronDetNetStage if knowledge else PlainDetNetStage
        self.stages = nn.ModuleList(
            [stage_type(input_dim, hidden, MIMO_N_TX) for _ in range(depth)]
        )
        self.step = nn.Parameter(torch.full((depth,), 0.35))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        H, y, sigma2 = batch["H"], batch["y"], batch["sigma2"]
        Ht = H.transpose(1, 2)
        hty = (Ht @ y.unsqueeze(-1)).squeeze(-1)
        prior = mmse_estimate(H, y, sigma2)
        x = torch.zeros_like(prior)
        for index, stage in enumerate(self.stages):
            residual = y - (H @ x.unsqueeze(-1)).squeeze(-1)
            gradient = (Ht @ residual.unsqueeze(-1)).squeeze(-1)
            if self.knowledge:
                features = torch.cat([x, gradient, hty, prior], dim=1)
                delta = stage(features, prior)
                x = torch.tanh(x + self.step[index].tanh() * delta + 0.12 * (prior - x))
            else:
                features = torch.cat([x, gradient, hty], dim=1)
                delta = stage(features)
                x = torch.tanh(x + self.step[index].tanh() * delta)
        return x


def complex_channel_batch(
    rng: np.random.Generator,
    n: int,
    taps: int,
) -> np.ndarray:
    power = np.exp(-np.arange(taps, dtype=np.float32) / max(1.0, taps * 0.35))
    power = power / power.sum()
    tap = (
        rng.standard_normal((n, taps)).astype(np.float32)
        + 1j * rng.standard_normal((n, taps)).astype(np.float32)
    ) * np.sqrt(power[None, :] / 2.0)
    padded = np.zeros((n, OFDM_K), dtype=np.complex64)
    padded[:, :taps] = tap
    h = np.fft.fft(padded, axis=1).astype(np.complex64)
    h /= np.sqrt(np.mean(np.abs(h) ** 2, axis=1, keepdims=True) + 1e-7)
    return h


def interpolate_pilots(h_pilot: np.ndarray, pilot_indices: np.ndarray) -> np.ndarray:
    grid = np.arange(OFDM_K)
    real = np.stack([np.interp(grid, pilot_indices, row.real) for row in h_pilot], axis=0)
    imag = np.stack([np.interp(grid, pilot_indices, row.imag) for row in h_pilot], axis=0)
    return (real + 1j * imag).astype(np.complex64)


def make_ofdm_data(
    n: int,
    snr_db: float | None,
    condition: str,
    seed: int,
) -> dict[str, Tensor]:
    rng = np.random.default_rng(seed)
    snr = np.full(n, 10.0 if snr_db is None else snr_db, dtype=np.float32)
    if snr_db is None:
        snr = rng.uniform(0.0, 20.0, size=n).astype(np.float32)
    taps = 8 if condition not in {"long_delay"} else 16
    h = complex_channel_batch(rng, n, taps)
    bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, OFDM_T, OFDM_K, 2))
    x = (bits[..., 0] + 1j * bits[..., 1]).astype(np.complex64) / np.sqrt(2.0)
    pilot_indices = np.arange(0, OFDM_K, OFDM_PILOT_SPACING)
    pilot_mask = np.zeros((OFDM_T, OFDM_K), dtype=np.float32)
    pilot_mask[0, pilot_indices] = 1.0
    pilot_value = np.complex64((1.0 + 1j) / np.sqrt(2.0))
    x[:, 0, pilot_indices] = pilot_value
    clean = h[:, None, :] * x
    noise_var = np.power(10.0, -snr / 10.0).astype(np.float32)
    noise = (
        rng.standard_normal(clean.shape).astype(np.float32)
        + 1j * rng.standard_normal(clean.shape).astype(np.float32)
    ) * np.sqrt(noise_var[:, None, None] / 2.0)
    if condition == "impulsive":
        outlier = rng.random(clean.shape) < 0.025
        noise += outlier * (
            rng.standard_normal(clean.shape).astype(np.float32)
            + 1j * rng.standard_normal(clean.shape).astype(np.float32)
        ) * np.sqrt(noise_var[:, None, None] / 2.0) * 8.0
    y = clean + noise
    if condition == "cfo":
        phase = np.exp(1j * 2.0 * np.pi * 0.09 * np.arange(OFDM_T, dtype=np.float32) / OFDM_T)
        y *= phase[None, :, None]
    if condition == "phase_noise":
        random_walk = np.cumsum(rng.standard_normal((n, OFDM_T)).astype(np.float32) * 0.035, axis=1)
        y *= np.exp(1j * random_walk[:, :, None])
    h_pilot = y[:, 0, pilot_indices] / pilot_value
    h_est = interpolate_pilots(h_pilot, pilot_indices)
    q = y / (h_est[:, None, :] + np.complex64(1e-2))
    confidence = np.clip(np.abs(h_est), 0.0, 3.0)[:, None, :]
    confidence = np.repeat(confidence, OFDM_T, axis=1)
    mask = np.repeat(pilot_mask[None, None, :, :], n, axis=0)
    features = np.stack(
        [
            np.clip(y.real, -4.0, 4.0),
            np.clip(y.imag, -4.0, 4.0),
            np.clip(q.real, -4.0, 4.0),
            np.clip(q.imag, -4.0, 4.0),
            confidence,
            mask[:, 0],
        ],
        axis=1,
    ).astype(np.float32)
    target = np.stack([x.real, x.imag], axis=1).astype(np.float32)
    data_mask = (1.0 - mask).astype(np.float32)
    return {
        "features": torch.from_numpy(features),
        "target": torch.from_numpy(target),
        "data_mask": torch.from_numpy(data_mask),
        "prior": torch.from_numpy(np.stack([q.real, q.imag], axis=1).astype(np.float32)),
    }


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = min(8, width)
        self.block = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(groups, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.block(x))


class DeepRxStyleCNN(nn.Module):
    """Parameter-rich fully convolutional receiver inspired by DeepRx."""

    def __init__(self, width: int = 64, blocks: int = 8):
        super().__init__()
        self.stem = nn.Conv2d(6, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2, 1),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        z = F.gelu(self.stem(batch["features"]))
        return self.head(self.blocks(z))


class KnowledgeNeuronReceiver(nn.Module):
    """Small residual receiver: q is a model-based prior, not a hard label."""

    def __init__(self, width: int = 16, blocks: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(6, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.delta_head = nn.Conv2d(width, 2, 1)
        self.trust_head = nn.Conv2d(width, 2, 1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        z = F.gelu(self.stem(batch["features"]))
        z = self.blocks(z)
        prior = batch["prior"]
        delta = torch.tanh(self.delta_head(z))
        trust = torch.sigmoid(self.trust_head(z))
        return torch.clamp(prior + trust * delta, -2.0, 2.0)


def model_specs() -> dict[str, dict[str, Any]]:
    return {
        "mimo": {
            "DetNet-L": {"kind": "plain", "depth": 12, "hidden": 128, "label": "Large DetNet-style NN"},
            "KNN-XS": {"kind": "knowledge", "depth": 4, "hidden": 32, "label": "Knowledge NN, 4 stages / 32 hidden"},
            "KNN-S": {"kind": "knowledge", "depth": 6, "hidden": 48, "label": "Knowledge NN, 6 stages / 48 hidden"},
            "KNN-M": {"kind": "knowledge", "depth": 8, "hidden": 64, "label": "Knowledge NN, 8 stages / 64 hidden"},
        },
        "ofdm": {
            "DeepRx-L": {"kind": "plain", "width": 64, "blocks": 8, "label": "Large DeepRx-style CNN"},
            "KNN-XS": {"kind": "knowledge", "width": 8, "blocks": 2, "label": "Knowledge CNN, width 8 / 2 blocks"},
            "KNN-S": {"kind": "knowledge", "width": 16, "blocks": 3, "label": "Knowledge CNN, width 16 / 3 blocks"},
            "KNN-M": {"kind": "knowledge", "width": 24, "blocks": 4, "label": "Knowledge CNN, width 24 / 4 blocks"},
        },
    }


def build_model(scenario: str, name: str) -> nn.Module:
    spec = model_specs()[scenario][name]
    if scenario == "mimo":
        return DetNetStyle(spec["depth"], spec["hidden"], spec["kind"] == "knowledge")
    if spec["kind"] == "knowledge":
        return KnowledgeNeuronReceiver(spec["width"], spec["blocks"])
    return DeepRxStyleCNN(spec["width"], spec["blocks"])


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_for(scenario: str, prediction: Tensor, batch: dict[str, Tensor]) -> Tensor:
    if scenario == "mimo":
        return F.mse_loss(prediction, batch["x"])
    mask = batch["data_mask"].expand_as(batch["target"])
    return ((prediction - batch["target"]) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


def train_model(
    scenario: str,
    model: nn.Module,
    train_data: dict[str, Tensor],
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    set_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3 if scenario == "mimo" else 2.0e-3, weight_decay=1e-5)
    n = next(iter(train_data.values())).shape[0]
    start = time.perf_counter()
    last_loss = float("nan")
    for _epoch in range(epochs):
        permutation = torch.randperm(n)
        for begin in range(0, n, batch_size):
            indices = permutation[begin : begin + batch_size]
            batch = move_batch({key: value[indices] for key, value in train_data.items()}, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = loss_for(scenario, prediction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    synchronize(device)
    return time.perf_counter() - start, last_loss


@torch.no_grad()
def evaluate_model(scenario: str, model: nn.Module | None, data: dict[str, Tensor], device: torch.device) -> dict[str, float]:
    n = next(iter(data.values())).shape[0]
    batch_size = 256
    total_bits = 0.0
    total_mse = 0.0
    total_count = 0.0
    if model is not None:
        model.eval()
    start = time.perf_counter()
    for begin in range(0, n, batch_size):
        batch_cpu = {key: value[begin : begin + batch_size] for key, value in data.items()}
        batch = move_batch(batch_cpu, device)
        if scenario == "mimo":
            prediction = mmse_estimate(batch["H"], batch["y"], batch["sigma2"]) if model is None else model(batch)
            target = batch["x"]
            total_bits += float((prediction.sign() != target.sign()).float().sum().cpu())
            total_mse += float(((prediction - target) ** 2).sum().cpu())
            total_count += float(target.numel())
        else:
            prediction = batch["prior"] if model is None else model(batch)
            target = batch["target"]
            mask = batch["data_mask"].expand_as(target)
            total_bits += float(((prediction.sign() != target.sign()).float() * mask).sum().cpu())
            total_mse += float((((prediction - target) ** 2) * mask).sum().cpu())
            total_count += float(mask.sum().cpu())
    synchronize(device)
    mse = total_mse / max(total_count, 1.0)
    return {
        "ber": total_bits / max(total_count, 1.0),
        "mse": mse,
        "evm": math.sqrt(mse),
        "inference_seconds": time.perf_counter() - start,
    }


def append_metric_rows(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
    condition: str,
    metrics: dict[str, float],
    params: int,
    train_seconds: float,
    last_loss: float | None,
) -> None:
    for metric in ("ber", "mse", "evm"):
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "condition": condition,
                "metric": metric,
                "value": float(metrics[metric]),
                "parameters": int(params),
                "train_seconds": float(train_seconds),
                "last_train_loss": None if last_loss is None else float(last_loss),
            }
        )


def add_robustness_summaries(rows: list[dict[str, Any]], scenario: str) -> None:
    methods = sorted({row["method"] for row in rows if row["scenario"] == scenario})
    conditions = set(SCENARIO_META[scenario]["conditions"])
    for method in methods:
        for metric in ("ber", "mse", "evm"):
            id_rows = [row for row in rows if row["scenario"] == scenario and row["method"] == method and row["condition"] == "id" and row["metric"] == metric]
            ood_rows = [row for row in rows if row["scenario"] == scenario and row["method"] == method and row["condition"] in conditions - {"id"} and row["metric"] == metric]
            if not id_rows or not ood_rows:
                continue
            base = id_rows[0]["value"]
            worst = max(row["value"] for row in ood_rows)
            ratio = worst / max(base, 1e-12) - 1.0
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "condition": "robustness_summary",
                    "metric": f"worst_{metric}",
                    "value": float(worst),
                    "parameters": id_rows[0]["parameters"],
                    "train_seconds": id_rows[0]["train_seconds"],
                    "last_train_loss": id_rows[0]["last_train_loss"],
                }
            )
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "condition": "robustness_summary",
                    "metric": f"robustness_gap_{metric}",
                    "value": float(ratio),
                    "parameters": id_rows[0]["parameters"],
                    "train_seconds": id_rows[0]["train_seconds"],
                    "last_train_loss": id_rows[0]["last_train_loss"],
                }
            )


def run_sample_sweep(
    scenario: str,
    sample_sizes: list[int],
    train_size: int,
    train_data_full: dict[str, Tensor],
    tests: dict[str, dict[str, Tensor]],
    main_models: dict[str, tuple[nn.Module, float, float]],
    sample_epochs: int,
    batch_size: int,
    device: torch.device,
    seed_base: int,
) -> list[dict[str, Any]]:
    names = ["DetNet-L", "KNN-M"] if scenario == "mimo" else ["DeepRx-L", "KNN-M"]
    robust_condition = SCENARIO_META[scenario]["robust_condition"]
    result: list[dict[str, Any]] = []
    for name in names:
        model, full_train_seconds, _full_loss = main_models[name]
        full_id = evaluate_model(scenario, model, tests["id"], device)
        full_robust = evaluate_model(scenario, model, tests[robust_condition], device)
        result.append(
            {
                "scenario": scenario,
                "method": name,
                "train_samples": train_size,
                "id_ber": full_id["ber"],
                "robust_ber": full_robust["ber"],
                "id_mse": full_id["mse"],
                "robust_mse": full_robust["mse"],
                "train_seconds": full_train_seconds,
                "parameters": parameter_count(model),
            }
        )
        for sample_size in sample_sizes:
            if sample_size >= train_size:
                continue
            if scenario == "mimo":
                sample_data = make_mimo_data(sample_size, None, "id", seed_base + sample_size)
            else:
                sample_data = make_ofdm_data(sample_size, None, "train", seed_base + sample_size)
            sample_model = build_model(scenario, name)
            seconds, _loss = train_model(scenario, sample_model, sample_data, sample_epochs, batch_size, device, seed_base + sample_size + 101)
            id_metrics = evaluate_model(scenario, sample_model, tests["id"], device)
            robust_metrics = evaluate_model(scenario, sample_model, tests[robust_condition], device)
            result.append(
                {
                    "scenario": scenario,
                    "method": name,
                    "train_samples": sample_size,
                    "id_ber": id_metrics["ber"],
                    "robust_ber": robust_metrics["ber"],
                    "id_mse": id_metrics["mse"],
                    "robust_mse": robust_metrics["mse"],
                    "train_seconds": seconds,
                    "parameters": parameter_count(sample_model),
                }
            )
    return result


def run_scenario(
    scenario: str,
    train_size: int,
    test_size: int,
    epochs: int,
    sample_sizes: list[int],
    sample_epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if scenario == "mimo":
        train_data = make_mimo_data(train_size, None, "id", seed)
        make_data: Callable[..., dict[str, Tensor]] = make_mimo_data
    else:
        train_data = make_ofdm_data(train_size, None, "train", seed)
        make_data = make_ofdm_data
    tests: dict[str, dict[str, Tensor]] = {}
    for index, condition in enumerate(SCENARIO_META[scenario]["conditions"]):
        condition_snr = 10.0
        if condition == "low_snr":
            condition_snr = 0.0
        elif condition == "high_snr":
            condition_snr = 20.0
        tests[condition] = make_data(test_size, condition_snr, condition, seed + 1000 + index)
    rows: list[dict[str, Any]] = []
    main_models: dict[str, tuple[nn.Module, float, float]] = {}
    for model_index, name in enumerate(model_specs()[scenario]):
        model = build_model(scenario, name)
        seconds, last_loss = train_model(scenario, model, train_data, epochs, batch_size, device, seed + model_index + 50)
        main_models[name] = (model, seconds, last_loss)
        params = parameter_count(model)
        for condition, data in tests.items():
            metrics = evaluate_model(scenario, model, data, device)
            append_metric_rows(rows, scenario, name, condition, metrics, params, seconds, last_loss)
    for condition, data in tests.items():
        baseline_metrics = evaluate_model(scenario, None, data, device)
        append_metric_rows(rows, scenario, "Model-based", condition, baseline_metrics, 0, 0.0, None)
    add_robustness_summaries(rows, scenario)
    sample_rows = run_sample_sweep(
        scenario,
        sample_sizes,
        train_size,
        train_data,
        tests,
        main_models,
        sample_epochs,
        batch_size,
        device,
        seed + 5000,
    )
    details = {
        "train_size": train_size,
        "test_size": test_size,
        "epochs": epochs,
        "sample_epochs": sample_epochs,
        "sample_sizes": sample_sizes,
        "robust_condition": SCENARIO_META[scenario]["robust_condition"],
        "models": {
            name: {
                "parameters": parameter_count(model),
                "train_seconds": seconds,
                "last_train_loss": last_loss,
                **model_specs()[scenario][name],
            }
            for name, (model, seconds, last_loss) in main_models.items()
        },
    }
    return rows, sample_rows, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for row_index, scenario in enumerate(("mimo", "ofdm")):
        methods = ["Model-based"] + list(model_specs()[scenario])
        colors = ["#6b7280"] + ["#2563eb"] + ["#f97316"] * 3
        for col_index, metric in enumerate(("ber", "mse")):
            ax = axes[row_index, col_index]
            values = []
            labels = []
            for method in methods:
                match = [r for r in rows if r["scenario"] == scenario and r["method"] == method and r["condition"] == "id" and r["metric"] == metric]
                if match:
                    values.append(match[0]["value"])
                    labels.append(method)
            ax.bar(labels, values, color=colors[: len(values)])
            ax.set_title(f"{scenario.upper()} in-distribution {metric.upper()}")
            ax.set_ylabel("lower is better")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
        if row_index == 0:
            ax = axes[row_index, 1]
    # Replace the lower-right panel with the parameter comparison shared by both scenarios.
    axes[1, 1].clear()
    for scenario, offset in (("mimo", -0.18), ("ofdm", 0.18)):
        names = list(model_specs()[scenario])
        params = [next(r["parameters"] for r in rows if r["scenario"] == scenario and r["method"] == name and r["condition"] == "id" and r["metric"] == "ber") for name in names]
        axes[1, 1].bar(np.arange(len(names)) + offset, params, width=0.35, label=scenario.upper())
    axes[1, 1].set_xticks(np.arange(4), ["large", "KNN-XS", "KNN-S", "KNN-M"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Trainable parameters (log scale)")
    axes[1, 1].set_ylabel("parameters")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle("Knowledge-neuron robustness and parameter-efficiency benchmark", fontsize=15)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def build_report(
    output: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    details: dict[str, Any],
) -> None:
    payload = json.dumps(
        {"summary": summary, "metrics": rows, "sample": sample_rows, "details": details, "meta": SCENARIO_META},
        ensure_ascii=False,
    ).replace("</script>", "<\\/script>")
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Knowledge-neuron robustness benchmark</title>
<style>
:root{--bg:#f6f8fb;--card:#fff;--ink:#172033;--muted:#5d687b;--line:#dbe2ec;--blue:#2563eb;--orange:#f97316;--gray:#6b7280;--green:#059669}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif;line-height:1.5}main{max-width:1240px;margin:0 auto;padding:28px}h1{margin:0 0 8px;font-size:30px}h2{margin-top:28px;font-size:21px}.lede{color:var(--muted);max-width:1000px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.stat,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 2px 8px #1720330b}.stat strong{display:block;font-size:20px}.stat span{font-size:12px;color:var(--muted)}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;background:var(--card);border:1px solid var(--line);padding:14px;border-radius:12px}.controls label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--muted)}select{min-width:190px;border:1px solid #b9c4d3;border-radius:7px;padding:8px;background:white;color:var(--ink)}.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px;font-size:13px}.legend span{padding:4px 8px;border-radius:999px;background:#eef2f8}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}.chart{min-height:160px;display:flex;flex-wrap:wrap;align-items:end;gap:14px;padding:18px 10px 8px;border-bottom:1px solid var(--line)}.bar{display:flex;flex-direction:column;justify-content:end;align-items:center;gap:4px;min-width:120px;flex:1}.bar-fill{width:100%;max-width:180px;border-radius:8px 8px 0 0;background:var(--blue);min-height:4px}.bar-fill.orange{background:var(--orange)}.bar-fill.gray{background:var(--gray)}.bar-fill.green{background:var(--green)}.bar small{color:var(--muted);text-align:center}.bar strong{font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.table-wrap{overflow:auto}.table-wrap table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card)}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}.note{border-left:4px solid var(--orange);background:#fff7ed;padding:12px 14px;border-radius:6px}.model-card{border-left:4px solid var(--blue);background:#eff6ff;padding:12px;border-radius:6px;margin:8px 0}.sample-line{display:grid;grid-template-columns:120px 120px 1fr 1fr;gap:8px;align-items:center;border-bottom:1px solid var(--line);padding:6px 0;font-size:13px}.sample-meter{height:10px;background:#edf1f6;border-radius:8px;overflow:hidden}.sample-meter i{display:block;height:100%;background:var(--orange)}code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}pre{background:#111827;color:#f3f4f6;padding:14px;border-radius:8px;overflow:auto}@media(max-width:760px){main{padding:16px}.stats,.grid{grid-template-columns:1fr}.sample-line{grid-template-columns:90px 90px 1fr}}
</style></head><body><main>
<h1>Knowledge-neuron robustness and parameter-efficiency benchmark</h1>
<p class="lede">Two paper-inspired wireless/DSP scenarios compare a parameter-rich neural baseline with smaller knowledge-neuron networks. The emphasis is not only one SNR point: the report also exposes parameter count, training time, sample efficiency, distribution-shift BER, MSE, EVM, and robustness gaps.</p>
<div class="stats"><div class="stat"><strong id="device"></strong><span>training device</span></div><div class="stat"><strong id="runtime"></strong><span>total benchmark time</span></div><div class="stat"><strong id="trainSize"></strong><span>samples per scenario</span></div><div class="stat"><strong id="torch"></strong><span>PyTorch / CUDA build</span></div></div>
<div class="controls"><label>Scenario<select id="scenario"></select></label><label>Condition<select id="condition"></select></label><label>Metric<select id="metric"></select></label></div>
<div class="card"><h2 id="scenarioTitle"></h2><p id="scenarioDescription" class="lede"></p><div id="chart" class="chart"></div><div id="table" class="table-wrap"></div></div>
<div class="grid"><div class="card"><h2>Model-size comparison</h2><div id="models" class="table-wrap"></div></div><div class="card"><h2>What the knowledge path changes</h2><div id="explanation"></div></div></div>
<div class="card"><h2>Sample efficiency</h2><p class="lede">The large baseline and KNN-M are retrained with smaller training sets. This is an empirical sample-efficiency curve, not a claim that one threshold is universal.</p><div id="sample"></div></div>
<div class="card"><h2>Interpretation notes</h2><div class="note">A knowledge-neuron model is expected to be most useful when a reliable physical prior reduces the search space or stabilizes a distribution shift. It is not expected to beat a correctly specified MMSE/FFT/matched-filter reference at every SNR. All metrics should be interpreted together with parameter count, wall-clock training time, and robustness gap.</div><h3>Reproduce</h3><pre>python run_robustness_benchmark.py --device auto
python run_robustness_benchmark.py --device cuda
python run_robustness_benchmark.py --quick</pre></div>
<script>const DATA=DATA_JSON;
const scenarioEl=document.querySelector('#scenario'),conditionEl=document.querySelector('#condition'),metricEl=document.querySelector('#metric');
const metricNames={mimo:['ber','mse','evm'],ofdm:['ber','mse','evm']};
const metricLabels={ber:'BER',mse:'MSE',evm:'EVM'};
const colorFor=(method)=>method==='Model-based'?'gray':method.startsWith('KNN')?'orange':'blue';
function fill(select,values,labels){select.innerHTML=values.map((v)=>'<option value="'+v+'">'+(labels&&labels[v]?labels[v]:v)+'</option>').join('');}
function updateConditions(){const s=scenarioEl.value;const values=Object.keys(DATA.meta[s].conditions);fill(conditionEl,values,Object.fromEntries(values.map((v)=>[v,DATA.meta[s].conditions[v]])));fill(metricEl,metricNames[s],metricLabels);render();}
function currentRows(){const s=scenarioEl.value,c=conditionEl.value,m=metricEl.value;return DATA.metrics.filter((r)=>r.scenario===s&&r.condition===c&&r.metric===m).sort((a,b)=>a.method.localeCompare(b.method));}
function render(){const s=scenarioEl.value,c=conditionEl.value,m=metricEl.value;const rows=currentRows();const max=Math.max(...rows.map((r)=>Number(r.value)),1e-12);document.querySelector('#scenarioTitle').textContent=DATA.meta[s].title;document.querySelector('#scenarioDescription').textContent=DATA.meta[s].description+' Selected condition: '+DATA.meta[s].conditions[c]+'.';document.querySelector('#chart').innerHTML=rows.map((r)=>'<div class="bar"><strong>'+Number(r.value).toPrecision(6)+'</strong><div class="bar-fill '+colorFor(r.method)+'" style="height:'+Math.max(5,100*Number(r.value)/max)+'px"></div><small>'+r.method+'</small></div>').join('');document.querySelector('#table').innerHTML='<table><thead><tr><th>Method</th><th>'+metricLabels[m]+'</th><th>Parameters</th><th>Train s</th></tr></thead><tbody>'+rows.map((r)=>'<tr><td>'+r.method+'</td><td>'+Number(r.value).toPrecision(8)+'</td><td>'+Number(r.parameters).toLocaleString()+'</td><td>'+Number(r.train_seconds).toFixed(2)+'</td></tr>').join('')+'</tbody></table>';renderModels(s);renderExplanation(s);renderSample(s);}
function renderModels(s){const entries=Object.entries(DATA.details[s].models);document.querySelector('#models').innerHTML='<table><thead><tr><th>Model</th><th>Parameters</th><th>Train s</th><th>Design</th></tr></thead><tbody>'+entries.map(([name,v])=>'<tr><td>'+name+'</td><td>'+Number(v.parameters).toLocaleString()+'</td><td>'+Number(v.train_seconds).toFixed(2)+'</td><td>'+v.label+'</td></tr>').join('')+'</tbody></table>';}
function renderExplanation(s){const meta=DATA.meta[s];const robust=meta.robust_condition;document.querySelector('#explanation').innerHTML='<div class="model-card"><strong>Paper-inspired large NN:</strong> '+meta.paper+' The large model is deliberately parameter-rich so that the comparison tests whether a smaller knowledge path can retain performance.</div><div class="model-card"><strong>Knowledge-neuron design:</strong> '+(s==='mimo'?'Each unfolded stage receives the MMSE estimate as a prior and adds a learned correction through a bounded trust gate.':'The model-based equalized symbol q is used as a residual prior; the compact CNN predicts a correction and a spatially varying trust mask.')+'</div><p><strong>Robustness condition:</strong> '+meta.conditions[robust]+'</p>'}
function renderSample(s){const rows=DATA.sample.filter((r)=>r.scenario===s);const max=Math.max(...rows.map((r)=>Number(r.id_ber)),1e-12);document.querySelector('#sample').innerHTML=rows.sort((a,b)=>a.method.localeCompare(b.method)||a.train_samples-b.train_samples).map((r)=>'<div class="sample-line"><strong>'+r.method+'</strong><span>'+Number(r.train_samples).toLocaleString()+' samples</span><span>ID BER '+Number(r.id_ber).toPrecision(5)+'<div class="sample-meter"><i style="width:'+Math.max(2,100*Number(r.id_ber)/max)+'%"></i></div></span><span>OOD BER '+Number(r.robust_ber).toPrecision(5)+'<div class="sample-meter"><i style="width:'+Math.max(2,100*Number(r.robust_ber)/max)+'%"></i></div></span></div>').join('');}
scenarioEl.addEventListener('change',updateConditions);conditionEl.addEventListener('change',render);metricEl.addEventListener('change',render);fill(scenarioEl,Object.keys(DATA.meta),{mimo:'MIMO detection',ofdm:'OFDM receiver'});scenarioEl.value='mimo';updateConditions();document.querySelector('#device').textContent=DATA.summary.device_name;document.querySelector('#runtime').textContent=Number(DATA.summary.total_runtime_seconds).toFixed(1)+' s';document.querySelector('#trainSize').textContent=Number(DATA.summary.train_size_per_scenario).toLocaleString();document.querySelector('#torch').textContent=DATA.summary.torch_version+' / '+DATA.summary.torch_cuda;</script></main></body></html>"""
    html = html.replace("DATA_JSON", payload)
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--sample-epochs", type=int, default=4)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 3
        args.train_size = 1024
        args.test_size = 256
        args.sample_epochs = 2
        args.sample_sizes = [256, 512]
    device = resolve_device(args.device)
    set_seed(SEED)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    started = time.perf_counter()
    for index, scenario in enumerate(("mimo", "ofdm")):
        rows, sample_rows, scenario_details = run_scenario(
            scenario,
            args.train_size,
            args.test_size,
            args.epochs,
            args.sample_sizes,
            args.sample_epochs,
            args.batch_size,
            device,
            SEED + index * 10000,
        )
        all_rows.extend(rows)
        all_sample_rows.extend(sample_rows)
        details[scenario] = scenario_details
    synchronize(device)
    total_seconds = time.perf_counter() - started
    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "train_size_per_scenario": args.train_size,
        "test_size_per_condition": args.test_size,
        "sample_epochs": args.sample_epochs,
        "sample_sizes": args.sample_sizes,
        "batch_size": args.batch_size,
        "seed": SEED,
        "total_runtime_seconds": total_seconds,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_csv(args.output_dir / "robustness_metrics.csv", all_rows)
    write_csv(args.output_dir / "sample_efficiency.csv", all_sample_rows)
    (args.output_dir / "robustness_summary.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2), encoding="utf-8")
    plot_summary(all_rows, args.output_dir / "robustness_summary.png")
    build_report(args.output_dir / "robustness_report.html", summary, all_rows, all_sample_rows, details)
    print(f"Device: {device} ({summary['device_name']})")
    print(f"PyTorch: {summary['torch_version']}; CUDA build: {summary['torch_cuda']}")
    print(f"Saved robustness report to {args.output_dir / 'robustness_report.html'}")
    print(f"Total runtime: {total_seconds:.2f} seconds")


if __name__ == "__main__":
    main()

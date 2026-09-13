"""Four-example wireless/DSP benchmark for model-based and knowledge neurons.

The script compares, for every task:

1. A traditional model-based reference.
2. A plain neural network.
3. A knowledge-neuron neural network with the same trainable backbone.

CUDA is selected automatically when available.  The reported comparison uses
task metrics (BER, channel NMSE, or frequency MAE); training losses are kept as
diagnostics and are intentionally not treated as comparable across tasks or
neuron types.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 19
N_TIME = 128
TARGET_INDEX = 64
BPSK_CHANNEL = np.array(
    [0.15 + 0.05j, 0.38 - 0.12j, 0.88 + 0.00j, 0.42 + 0.11j, 0.18 - 0.04j],
    dtype=np.complex64,
)
BPSK_CHANNEL = BPSK_CHANNEL / np.sqrt(np.mean(np.abs(BPSK_CHANNEL) ** 2))
OFDM_N = 64
OFDM_PILOTS = np.arange(0, OFDM_N, 4)
FREQ_MIN = 0.04
FREQ_MAX = 0.36


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device("cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu")


def awgn(signal: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    power = np.mean(np.abs(signal) ** 2, axis=tuple(range(1, signal.ndim)))
    noise_power = power / np.power(10.0, snr_db / 10.0)
    noise = rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    return signal + noise * np.sqrt(noise_power.reshape((-1,) + (1,) * (signal.ndim - 1)) / 2.0)


def make_bpsk_dataset(
    n: int,
    snr_db: float | np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, N_TIME))
    perturbation = (
        rng.normal(size=(n, BPSK_CHANNEL.size))
        + 1j * rng.normal(size=(n, BPSK_CHANNEL.size))
    ).astype(np.complex64)
    h_true = BPSK_CHANNEL[None, :] * (1.0 + jitter * perturbation / np.sqrt(2.0))
    h_true *= np.sqrt(np.mean(np.abs(BPSK_CHANNEL) ** 2)) / np.sqrt(
        np.mean(np.abs(h_true) ** 2, axis=1, keepdims=True)
    )
    received = np.zeros((n, N_TIME), dtype=np.complex64)
    for tap, coefficient in enumerate(h_true.T):
        received[:, tap:] += coefficient[:, None] * symbols[:, : N_TIME - tap]
    snr_values = np.broadcast_to(np.asarray(snr_db, dtype=np.float32), (n,))
    received = awgn(received, snr_values, rng).astype(np.complex64)

    window = received[:, TARGET_INDEX : TARGET_INDEX + BPSK_CHANNEL.size]
    matched_filter = np.real(np.sum(np.conj(BPSK_CHANNEL[None, :]) * window, axis=1))
    matched_filter /= np.sum(np.abs(BPSK_CHANNEL) ** 2)
    features = np.stack([received.real, received.imag], axis=1).astype(np.float32)
    labels = (symbols[:, TARGET_INDEX] > 0).astype(np.float32).reshape(-1, 1)
    return features, matched_filter.astype(np.float32).reshape(-1, 1), labels


def make_ofdm_dataset(
    n: int,
    snr_db: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    taps = 8
    tap_decay = np.exp(-np.arange(taps, dtype=np.float32) / 3.0)
    h_time = (
        rng.normal(size=(n, taps)) + 1j * rng.normal(size=(n, taps))
    ).astype(np.complex64) * tap_decay[None, :]
    h_time /= np.sqrt(np.mean(np.abs(h_time) ** 2, axis=1, keepdims=True))
    h_pad = np.zeros((n, OFDM_N), dtype=np.complex64)
    h_pad[:, :taps] = h_time
    channel = np.fft.fft(h_pad, axis=1).astype(np.complex64)

    snr_values = np.broadcast_to(np.asarray(snr_db, dtype=np.float32), (n,))
    pilot_clean = channel[:, OFDM_PILOTS]
    pilot_power = np.mean(np.abs(pilot_clean) ** 2, axis=1)
    pilot_noise_power = pilot_power / np.power(10.0, snr_values / 10.0)
    noise = (
        rng.normal(size=pilot_clean.shape) + 1j * rng.normal(size=pilot_clean.shape)
    ) * np.sqrt(pilot_noise_power[:, None] / 2.0)
    pilot_ls = pilot_clean + noise.astype(np.complex64)

    sparse = np.zeros((n, OFDM_N), dtype=np.complex64)
    sparse[:, OFDM_PILOTS] = pilot_ls
    mask = np.zeros(OFDM_N, dtype=np.float32)
    mask[OFDM_PILOTS] = 1.0
    baseline_real = np.stack(
        [np.interp(np.arange(OFDM_N), OFDM_PILOTS, row.real) for row in pilot_ls], axis=0
    )
    baseline_imag = np.stack(
        [np.interp(np.arange(OFDM_N), OFDM_PILOTS, row.imag) for row in pilot_ls], axis=0
    )
    baseline = (baseline_real + 1j * baseline_imag).astype(np.complex64)
    features = np.stack([sparse.real, sparse.imag, np.broadcast_to(mask, (n, OFDM_N))], axis=-1)
    target = np.stack([channel.real, channel.imag], axis=-1).astype(np.float32)
    baseline_ri = np.stack([baseline.real, baseline.imag], axis=-1).astype(np.float32)
    return features.astype(np.float32), baseline_ri, target, channel


def fft_frequency_estimate(signal: np.ndarray) -> np.ndarray:
    n_fft = 1024
    window = np.hanning(signal.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(signal * window[None, :], n=n_fft, axis=1))
    peak = np.argmax(spectrum[:, 1:-1], axis=1) + 1
    left = np.take_along_axis(spectrum, (peak - 1)[:, None], axis=1)[:, 0]
    middle = np.take_along_axis(spectrum, peak[:, None], axis=1)[:, 0]
    right = np.take_along_axis(spectrum, (peak + 1)[:, None], axis=1)[:, 0]
    denominator = left - 2.0 * middle + right
    delta = 0.5 * (left - right) / np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
    return ((peak + delta) / n_fft).astype(np.float32)


def make_tone_dataset(
    n: int,
    snr_db: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_axis = np.arange(N_TIME, dtype=np.float32)[None, :]
    frequency = rng.uniform(FREQ_MIN, FREQ_MAX, size=n).astype(np.float32)
    amplitude = rng.uniform(0.8, 1.2, size=n).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    clean = amplitude[:, None] * np.sin(
        2.0 * np.pi * frequency[:, None] * time_axis + phase[:, None]
    )
    snr_values = np.broadcast_to(np.asarray(snr_db, dtype=np.float32), (n,))
    signal_power = np.mean(clean**2, axis=1)
    noise_power = signal_power / np.power(10.0, snr_values / 10.0)
    noisy = clean + rng.normal(size=clean.shape) * np.sqrt(noise_power[:, None])
    fft_baseline = fft_frequency_estimate(noisy)
    return (
        noisy.astype(np.float32)[..., None],
        fft_baseline.reshape(-1, 1).astype(np.float32),
        frequency.reshape(-1, 1),
        clean.astype(np.float32),
    )


def make_mimo_dataset(
    n: int,
    snr_db: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 2x2 flat-fading QPSK detection problem."""
    bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, 2, 2))
    symbols = (bits[:, :, 0] + 1j * bits[:, :, 1]) / np.sqrt(2.0)
    channel = (
        rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
    ).astype(np.complex64) / np.sqrt(2.0)
    received_clean = np.einsum("nij,nj->ni", channel, symbols)
    snr_values = np.broadcast_to(np.asarray(snr_db, dtype=np.float32), (n,))
    received = awgn(received_clean, snr_values, rng).astype(np.complex64)

    signal_power = np.mean(np.abs(received_clean) ** 2, axis=1)
    noise_variance = signal_power / np.power(10.0, snr_values / 10.0)
    h_h = np.einsum("nki,nkj->nij", np.conj(channel), channel)
    h_y = np.einsum("nki,nk->ni", np.conj(channel), received)
    regularizer = noise_variance[:, None, None] * np.eye(2, dtype=np.complex64)[None, :, :]
    mmse = np.linalg.solve(h_h + regularizer, h_y[..., None])[..., 0]
    soft_prior = np.stack([mmse.real, mmse.imag], axis=-1).reshape(n, 4).astype(np.float32)
    features = np.concatenate(
        [
            np.stack([received.real, received.imag], axis=-1).reshape(n, 4),
            np.stack([channel.real, channel.imag], axis=-1).reshape(n, 8),
        ],
        axis=1,
    ).astype(np.float32)
    targets = (bits > 0).astype(np.float32).reshape(n, 4)
    return features, soft_prior, targets


class KnowledgeReLU(nn.Module):
    """ReLU with a learnable trust gate for a bounded knowledge residual."""

    def __init__(self, strength: float = 0.12) -> None:
        super().__init__()
        self.max_strength = strength
        self.trust_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: Tensor, knowledge: Tensor) -> Tensor:
        k = torch.tanh(knowledge)
        while k.ndim < z.ndim:
            k = k.unsqueeze(-1)
        if k.shape[-1] != 1:
            k = k.mean(dim=-1, keepdim=True)
        trust = self.max_strength * torch.sigmoid(self.trust_logit)
        return F.relu(z) + trust * k


class KnowledgeGELU(nn.Module):
    """GELU with the same learnable trust gate."""

    def __init__(self, strength: float = 0.10) -> None:
        super().__init__()
        self.max_strength = strength
        self.trust_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: Tensor, knowledge: Tensor) -> Tensor:
        k = torch.tanh(knowledge)
        while k.ndim < z.ndim:
            k = k.unsqueeze(-1)
        if k.shape[-1] != 1:
            k = k.mean(dim=-1, keepdim=True)
        trust = self.max_strength * torch.sigmoid(self.trust_logit)
        return F.gelu(z) + trust * k


class BPSKCNN(nn.Module):
    def __init__(self, knowledge: bool) -> None:
        super().__init__()
        self.knowledge = knowledge
        self.conv1 = nn.Conv1d(2, 48, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(48, 96, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(96, 96, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(96, 48, kernel_size=5, padding=2)
        self.head = nn.Linear(48, 1)
        self.krelu = KnowledgeReLU()

    def forward(self, x: Tensor, knowledge: Tensor) -> Tensor:
        if self.knowledge:
            h = self.krelu(self.conv1(x), knowledge)
            h = self.krelu(self.conv2(h), knowledge)
            h = self.krelu(self.conv3(h), knowledge)
            h = self.krelu(self.conv4(h), knowledge)
        else:
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            h = F.relu(self.conv4(h))
        return self.head(h[:, :, TARGET_INDEX])


class KnowledgeTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 48, n_heads: int = 4, feedforward: int = 96) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.linear1 = nn.Linear(d_model, feedforward)
        self.linear2 = nn.Linear(feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.kgelu = KnowledgeGELU()

    def forward(self, x: Tensor, knowledge: Tensor) -> Tensor:
        attended = self.attention(x, x, x, need_weights=False)[0]
        x = self.norm1(x + attended)
        feedforward = self.linear2(self.kgelu(self.linear1(x), knowledge))
        return self.norm2(x + feedforward)


class OFDMTransformer(nn.Module):
    def __init__(self, knowledge: bool, layers: int = 3, d_model: int = 48) -> None:
        super().__init__()
        self.knowledge = knowledge
        self.input = nn.Linear(3, d_model)
        self.position = nn.Parameter(torch.zeros(1, OFDM_N, d_model))
        if knowledge:
            self.encoder = nn.ModuleList(
                [KnowledgeTransformerLayer(d_model=d_model) for _ in range(layers)]
            )
        else:
            self.encoder = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=4,
                        dim_feedforward=64,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(layers)
                ]
            )
        self.output = nn.Linear(d_model, 2)
        self.residual_trust = nn.Parameter(torch.tensor(0.0)) if knowledge else None

    def forward(self, x: Tensor, knowledge: Tensor) -> Tensor:
        h = self.input(x) + self.position
        for layer in self.encoder:
            h = layer(h, knowledge) if self.knowledge else layer(h)
        prediction = self.output(h)
        # Model-informed residual learning: start from linear pilot interpolation.
        if self.knowledge:
            residual_scale = 0.5 * torch.sigmoid(self.residual_trust)
            return knowledge + residual_scale * prediction
        return prediction


class FrequencyGRU(nn.Module):
    def __init__(self, knowledge: bool) -> None:
        super().__init__()
        self.knowledge = knowledge
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(64, 48)
        self.linear2 = nn.Linear(48, 1)
        self.krelu = KnowledgeReLU(strength=0.18)
        self.residual_trust = nn.Parameter(torch.tensor(0.0)) if knowledge else None

    def forward(self, x: Tensor, knowledge: Tensor) -> Tensor:
        hidden = self.gru(x)[0][:, -1, :]
        if self.knowledge:
            correction = self.linear2(self.krelu(self.linear1(hidden), knowledge))
            residual_scale = 0.08 * torch.sigmoid(self.residual_trust)
            return torch.clamp(knowledge + residual_scale * torch.tanh(correction), FREQ_MIN, FREQ_MAX)
        raw = self.linear2(F.relu(self.linear1(hidden)))
        return FREQ_MIN + (FREQ_MAX - FREQ_MIN) * torch.sigmoid(raw)


class MIMOMLP(nn.Module):
    """A wider MLP for 2x2 QPSK detection with an MMSE knowledge path."""

    def __init__(self, knowledge: bool) -> None:
        super().__init__()
        self.knowledge = knowledge
        self.linear1 = nn.Linear(12, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 4)
        self.krelu = KnowledgeReLU(strength=0.20)
        self.output_trust = nn.Parameter(torch.tensor(0.0)) if knowledge else None

    def forward(self, x: Tensor, knowledge: Tensor) -> Tensor:
        if self.knowledge:
            h = self.krelu(self.linear1(x), knowledge)
            h = self.krelu(self.linear2(h), knowledge)
            h = self.krelu(self.linear3(h), knowledge)
            trust = 0.75 * torch.sigmoid(self.output_trust)
            return self.output(h) + trust * knowledge
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        h = F.relu(self.linear3(h))
        return self.output(h)


@torch.no_grad()
def task_metric(task: str, prediction: Tensor, target: Tensor) -> float:
    if task == "bpsk":
        return float(((prediction >= 0).float() != target).float().mean().item())
    if task == "mimo":
        return float(((prediction >= 0).float() != target).float().mean().item())
    if task == "ofdm":
        error = torch.sum((prediction - target) ** 2, dim=(1, 2))
        energy = torch.sum(target**2, dim=(1, 2)).clamp_min(1e-8)
        return float((error / energy).mean().item())
    if task == "tone":
        return float(torch.abs(prediction - target).mean().item())
    raise ValueError(task)


def train_model(
    task: str,
    model: nn.Module,
    name: str,
    train_x: Tensor,
    train_knowledge: Tensor,
    train_y: Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, object]]:
    model.to(device)
    loader = DataLoader(
        TensorDataset(train_x, train_knowledge, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() if task in {"bpsk", "mimo"} else nn.MSELoss()
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for xb, kb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            kb = kb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb, kb)
            loss = criterion(prediction, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
        history.append(
            {
                "task": task,
                "model": name,
                "epoch": epoch,
                "loss": total_loss / max(count, 1),
            }
        )
    return history


@torch.no_grad()
def predict_metric(
    task: str,
    model: nn.Module,
    x: Tensor,
    knowledge: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    model.eval()
    prediction = model(x.to(device), knowledge.to(device))
    return task_metric(task, prediction, y.to(device))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_benchmark(output_dir: Path, metrics: list[dict[str, object]]) -> None:
    examples = [
        ("bpsk", "BPSK multipath detection", "BER", True),
        ("ofdm", "OFDM channel estimation", "NMSE", True),
        ("tone", "DSP tone frequency estimation", "MAE (cycles/sample)", False),
        ("mimo", "2x2 MIMO QPSK detection", "BER", True),
    ]
    colors = {
        "Model-based": "#6b7280",
        "Plain NN": "#4c78a8",
        "Knowledge-neuron NN": "#f58518",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.ravel()
    for axis, (task, title, ylabel, log_y) in zip(axes, examples):
        task_rows = [row for row in metrics if row["task"] == task]
        for method, color in colors.items():
            rows = [row for row in task_rows if row["method"] == method]
            axis.plot(
                [row["snr_db"] for row in rows],
                [row["value"] for row in rows],
                marker="o",
                linewidth=2,
                label=method,
                color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel(ylabel + (" (log scale; lower is better)" if log_y else " (lower is better)"))
        if log_y:
            axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(output_dir / "benchmark_metrics.png", dpi=170)
    plt.close(fig)


def build_report(
    output_dir: Path,
    summary: dict[str, object],
    metrics: list[dict[str, object]],
    histories: list[dict[str, object]],
) -> None:
    descriptions = {
        "bpsk": (
            "输入 128 个复数接收样本，目标是检测中心 BPSK 符号。"
            "Model-based 方法是已知名义信道下的匹配滤波判决；NN 使用 1D CNN；"
            "knowledge-neuron NN 在 CNN 中注入匹配滤波分数。"
        ),
        "ofdm": (
            "输入 64 个子载波上的稀疏导频 LS 估计和 pilot mask，目标是恢复完整频域信道。"
            "Model-based 方法是实部/虚部线性插值；NN 使用小型 Transformer；"
            "knowledge-neuron NN 以插值结果作为 residual 起点，并在 Transformer FFN 中使用 K-GELU。"
        ),
        "tone": (
            "输入带噪单音序列，目标是估计归一化频率。Model-based 方法是零填充 FFT 峰值估计；"
            "NN 使用 GRU；knowledge-neuron NN 用 FFT 估计作为输出神经元的物理先验，并学习小幅 residual。"
        ),
    }
    task_names = {
        "bpsk": "例子 1：多径 BPSK 检测",
        "ofdm": "例子 2：OFDM 信道估计",
        "tone": "例子 3：DSP 单音频率估计",
    }
    task_metric_names = {"bpsk": "BER", "ofdm": "NMSE", "tone": "frequency MAE"}
    sections = []
    for task in ["bpsk", "ofdm", "tone"]:
        rows = [row for row in metrics if row["task"] == task]
        snrs = sorted({float(row["snr_db"]) for row in rows})
        body = []
        for snr in snrs:
            values = []
            for method in ["Model-based", "Plain NN", "Knowledge-neuron NN"]:
                value = next(
                    float(row["value"])
                    for row in rows
                    if row["method"] == method and float(row["snr_db"]) == snr
                )
                values.append(f"<td>{value:.6f}</td>")
            body.append(f"<tr><td>{snr:g}</td>{''.join(values)}</tr>")
        sections.append(
            f"""<h2>{task_names[task]}</h2>
<p>{descriptions[task]}</p>
<table><thead><tr><th>SNR (dB)</th><th>Model-based</th><th>Plain NN</th><th>Knowledge-neuron NN</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p><small>指标：{task_metric_names[task]}，数值越低越好。完整数据在 <a href="benchmark_metrics.csv">benchmark_metrics.csv</a>。</small></p>"""
        )

    task_history = []
    for task in ["bpsk", "ofdm", "tone"]:
        grouped = {}
        for row in histories:
            if row["task"] == task:
                grouped.setdefault(row["model"], []).append(row)
        for model, rows in grouped.items():
            task_history.append(
                f"<li>{task}: {model} 最终训练 loss = {float(rows[-1]['loss']):.6f}</li>"
            )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wireless/DSP knowledge-neuron benchmark</title>
<style>
:root {{ color-scheme:light; --ink:#18202a; --muted:#5d6875; --line:#d9dee5; --accent:#1f6feb; --soft:#f5f7fa; }}
body {{ margin:0; color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
main {{ max-width:1150px; margin:0 auto; padding:42px 24px 72px; }}
h1 {{ font-size:30px; line-height:1.2; margin:0 0 10px; }} h2 {{ margin-top:38px; font-size:21px; }}
.lede {{ color:var(--muted); font-size:16px; max-width:900px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:24px 0; }}
.stat {{ background:var(--soft); border:1px solid var(--line); padding:14px 16px; }} .stat strong {{ display:block; font-size:21px; }} .stat span {{ color:var(--muted); font-size:13px; }}
img {{ max-width:100%; border:1px solid var(--line); }} table {{ border-collapse:collapse; width:100%; font-size:14px; }} th,td {{ border-bottom:1px solid var(--line); text-align:right; padding:8px; }} th:first-child,td:first-child {{ text-align:left; }}
pre {{ background:#111827; color:#e5e7eb; overflow:auto; padding:14px; }} code,pre {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }} code {{ background:var(--soft); padding:2px 4px; }}
.note {{ border-left:3px solid var(--accent); padding:8px 14px; background:#f7fbff; }} a {{ color:var(--accent); }} small {{ color:var(--muted); }}
</style></head><body><main>
<h1>Wireless/DSP model-based vs knowledge-neuron benchmark</h1>
<p class="lede">四个任务、三种方法、四种不同架构。结果由本机自动运行生成，并使用各任务的最终指标比较。</p>
<div class="grid">
<div class="stat"><strong>{summary["device_name"]}</strong><span>训练设备</span></div>
<div class="stat"><strong>{summary["torch_version"]}</strong><span>PyTorch / CUDA: {summary["torch_cuda"]}</span></div>
<div class="stat"><strong>{summary["total_runtime_seconds"]:.1f} s</strong><span>四例总运行时间</span></div>
<div class="stat"><strong>{summary["total_trainable_params"]:,}</strong><span>四例 NN 参数总量</span></div>
</div>
<img src="benchmark_metrics.png" alt="三种方法在四个无线/DSP任务上的误差指标">
{''.join(sections)}
<h2>为什么不横向比较 loss</h2>
<div class="note">这里把 BER、信道 NMSE 和频率 MAE 作为最终任务指标。不同任务的损失函数、输出尺度、数据噪声和知识残差结构不同；即使是同一任务，knowledge-neuron 的 loss 高于 plain NN，也不能单独推出性能更差。loss 只作为训练诊断保留：<a href="benchmark_history.csv">benchmark_history.csv</a>。</div>
<ul>{''.join(task_history)}</ul>
<h2>如何运行</h2>
<pre>python -m pip install -r requirements.txt
python multi_example_benchmark.py --device auto

# 强制 GPU
python multi_example_benchmark.py --device cuda

# 快速 smoke test
python multi_example_benchmark.py --quick</pre>
<p><code>--device auto</code> 会优先使用 CUDA；脚本会把训练数据分批搬到 GPU，并记录实际设备。另一台电脑无需 MATLAB，也无需向 GitHub push，只要 pull 后本地运行即可。</p>
<h2>复现实验与硬件判断</h2>
<p>本次 benchmark 的数据规模刻意保持在研究迭代友好的范围。总耗时为 {summary["total_runtime_seconds"]:.1f} 秒，因此当前规模没有必要为了速度额外打包迁移。你的 RTX PRO 500 Blackwell Laptop GPU 理论上可用 CUDA；但该 benchmark 很小，GPU kernel 启动和数据搬运开销可能占比较高，速度不一定按硬件规格线性提升。若把样本数、Transformer 层数或 Monte-Carlo 重复次数扩大，远程机器更可能体现优势。</p>
<h2>参考</h2>
<ul>
<li><a href="https://arxiv.org/abs/1702.00832">O'Shea &amp; Hoydis, An Introduction to Deep Learning for the Physical Layer</a></li>
<li><a href="https://pytorch.org/get-started/locally/">PyTorch official installation selector</a></li>
<li><a href="https://developer.nvidia.com/cuda/gpus">NVIDIA CUDA GPU compute capability list</a></li>
</ul>
<p><small>生成时间：{summary["generated_at"]}</small></p>
</main></body></html>"""
    (output_dir / "benchmark_report.html").write_text(html, encoding="utf-8")


def build_interactive_report(
    output_dir: Path,
    summary: dict[str, object],
    metrics: list[dict[str, object]],
    histories: list[dict[str, object]],
) -> None:
    task_order = ["bpsk", "ofdm", "tone", "mimo"]
    task_meta = {
        "bpsk": {
            "title": "例子 1：多径 BPSK 检测",
            "metric": "BER",
            "unit": "bit error rate",
            "description": "从 128 个复数接收样本中检测中心 BPSK 符号。真实 5-tap 信道围绕名义信道随机扰动。",
            "model": "已知名义信道的匹配滤波 + sign 判决。它是一个很强的专家基线，但信道失配和 ISI 会限制它。",
            "plain": "1D CNN：2 个输入通道（I/Q），经过 2→48→96→96→48 的四层卷积，读取中心位置的 logit。",
            "knowledge": "相同 CNN backbone；每个隐藏卷积后使用 K-ReLU，并把匹配滤波分数 q 作为 bounded residual。每个 K-ReLU 学一个 trust gate。",
            "architecture": [
                {"label": "Model-based", "steps": ["I/Q waveform", "5-tap matched filter", "sign → bit"], "note": "不训练 NN"},
                {"label": "Plain NN", "steps": ["2×128 I/Q", "Conv 2→48→96→96→48", "center logit"], "note": "ReLU"},
                {"label": "Knowledge-neuron NN", "steps": ["2×128 I/Q", "Conv + K-ReLU ×4", "center logit + q residual"], "note": "learnable trust"},
            ],
            "why": "当 CNN 有足够数据时，它会自己学出局部相关、匹配滤波和抗 ISI 表示，因此 q 不一定再带来很大增益；如果名义信道失配，q 还可能成为有偏输入。",
        },
        "ofdm": {
            "title": "例子 2：OFDM 信道估计",
            "metric": "NMSE",
            "unit": "normalized mean square error",
            "description": "从 64 个子载波的稀疏导频 LS 估计和 pilot mask 恢复完整频域信道。",
            "model": "实部和虚部分别做线性插值。它利用了频域信道平滑性，但不知道真实多径统计。",
            "plain": "Transformer encoder：64 个频域 token，3→48 embedding，3 层 4-head self-attention，FFN 48→96→48，输出每个子载波的复数信道。",
            "knowledge": "相同 Transformer；K-GELU 读取插值信道作为先验，输出学习 residual，并通过 learnable residual trust 决定相信 baseline 多少。",
            "architecture": [
                {"label": "Model-based", "steps": ["pilot LS + mask", "linear interpolation", "Ĥ[k]"], "note": "频域平滑先验"},
                {"label": "Plain NN", "steps": ["64 tokens × 3", "3× Transformer d=48", "64×2 channel"], "note": "GELU"},
                {"label": "Knowledge-neuron NN", "steps": ["64 tokens × 3", "3× K-GELU Transformer", "Ĥinterp + learned residual"], "note": "adaptive trust"},
            ],
            "why": "在高 SNR 时线性插值已经很接近真实信道，神经网络没有太多可修正的误差；在低 SNR 或信道频率选择性明显时，学习 residual 才可能明显超过传统插值。",
        },
        "tone": {
            "title": "例子 3：DSP 单音频率估计",
            "metric": "frequency MAE",
            "unit": "cycles/sample",
            "description": "从长度 128 的带噪单音序列估计归一化频率。",
            "model": "Hann window + 1024-point zero-padded FFT peak interpolation。",
            "plain": "GRU：2 层、hidden size 64，最后 hidden state 经过 64→48→1 head 输出频率。",
            "knowledge": "相同 GRU；K-ReLU 使用 FFT 峰值作为先验，输出为 q 加一个受 trust gate 限制的 learned correction。",
            "architecture": [
                {"label": "Model-based", "steps": ["noisy tone", "Hann + 1024 FFT", "parabolic peak"], "note": "强解析基线"},
                {"label": "Plain NN", "steps": ["128×1 sequence", "2-layer GRU h=64", "64→48→1"], "note": "ReLU"},
                {"label": "Knowledge-neuron NN", "steps": ["128×1 sequence", "2-layer GRU + K-ReLU", "qFFT + learned correction"], "note": "bounded residual"},
            ],
            "why": "这个任务的 FFT baseline 已经非常强；knowledge neuron 的合理目标不是把它彻底替换，而是学习小幅修正。此时 plain NN 反而可能因为有限数据而不如解析方法。",
        },
        "mimo": {
            "title": "例子 4：2×2 MIMO QPSK 检测",
            "metric": "BER",
            "unit": "bit error rate",
            "description": "从 2×2 平坦 Rayleigh MIMO 的接收向量 y 和信道矩阵 H 恢复两个 QPSK 流的 4 个 bit。",
            "model": "线性 MMSE 检测器：x̂=(HᴴH+σ²I)⁻¹Hᴴy，再对实部和虚部 sign 判决。",
            "plain": "宽 MLP：12 个输入（y 的 I/Q 加 H 的 I/Q），128→128→64→4，普通 ReLU。",
            "knowledge": "相同 MLP；K-ReLU 读取 MMSE soft estimate，输出 logit 再加 learnable trust×MMSE prior。",
            "architecture": [
                {"label": "Model-based", "steps": ["y, H, σ²", "linear MMSE solve", "sign → 4 bits"], "note": "矩阵模型"},
                {"label": "Plain NN", "steps": ["12 features", "128→128→64", "4 logits"], "note": "ReLU"},
                {"label": "Knowledge-neuron NN", "steps": ["12 features", "128→128→64 + K-ReLU", "4 logits + MMSE prior"], "note": "adaptive trust"},
            ],
            "why": "当 H 和噪声模型准确时，MMSE 已经是强基线；NN 只有在非理想信道、模型失配或需要学习额外统计时才有明显空间。knowledge path 的价值通常是稳定收敛和降低数据需求，而不是保证每个 SNR 点都获胜。",
        },
    }
    payload = {
        "summary": summary,
        "metrics": metrics,
        "histories": histories,
        "taskOrder": task_order,
        "taskMeta": task_meta,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive wireless/DSP knowledge-neuron benchmark</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#5b6673;--line:#d8dee6;--soft:#f5f7fa;--blue:#3f6fa6;--orange:#e07820;--gray:#737b84;--accent:#1f6feb}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}
main{max-width:1180px;margin:0 auto;padding:34px 24px 70px}h1{font-size:30px;line-height:1.18;margin:0 0 10px}h2{font-size:21px;margin:34px 0 10px}h3{font-size:16px;margin:20px 0 6px}.lede{color:var(--muted);max-width:920px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:22px 0}.stat{background:var(--soft);border:1px solid var(--line);padding:13px 15px}.stat strong{display:block;font-size:20px}.stat span{color:var(--muted);font-size:12px}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:14px;margin:22px 0 14px;padding-bottom:12px;border-bottom:1px solid var(--line)}label{font-size:13px;color:var(--muted);display:flex;flex-direction:column;gap:4px}select,button{font:inherit}select{border:1px solid var(--line);padding:8px 10px;background:#fff;color:var(--ink)}button{border:1px solid var(--line);background:#fff;padding:8px 11px;cursor:pointer}button:hover,button:focus-visible{border-color:var(--accent);outline:2px solid #b9d3f5;outline-offset:1px}
.method-toggle{display:flex;flex-direction:row;align-items:center;gap:5px;color:var(--ink)}.method-toggle input{accent-color:var(--accent)}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%}.blue{background:var(--blue)}.orange{background:var(--orange)}.gray{background:var(--gray)}
.chart-wrap{border:1px solid var(--line);padding:8px 8px 0}.chart-wrap svg{width:100%;height:auto;display:block}.axis{stroke:#8a939c;stroke-width:1}.gridline{stroke:#e4e8ed;stroke-width:1}.series{fill:none;stroke-width:3}.point{stroke:#fff;stroke-width:1.5;cursor:pointer}.axis-label{fill:var(--muted);font-size:12px}.chart-title{fill:var(--ink);font-size:15px;font-weight:500}.chart-legend{font-size:12px;fill:var(--ink)}
.snapshot{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0}.bar-item{border-bottom:1px solid var(--line);padding:8px 0}.bar-label{display:flex;justify-content:space-between;font-size:13px}.bar-track{height:10px;background:#eef1f4;margin-top:5px}.bar-fill{height:100%}.bar-fill.blue{background:var(--blue)}.bar-fill.orange{background:var(--orange)}.bar-fill.gray{background:var(--gray)}
.architecture{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.arch-col{border-top:2px solid var(--line);padding-top:8px}.arch-col h3{margin-top:0}.arch-step{display:flex;align-items:center;gap:5px;min-height:36px;border:1px solid var(--line);padding:6px 8px;margin:6px 0;background:#fafbfc;font-size:13px}.arch-step:not(:last-child)::after{content:"↓";position:absolute;transform:translateY(25px);color:var(--muted)}.arch-note{font-size:12px;color:var(--muted)}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.explain{border-left:3px solid var(--accent);padding:10px 14px;background:#f7fbff}.details{border-top:1px solid var(--line);margin-top:14px}.details summary{cursor:pointer;padding:10px 0;font-weight:500}.details p{margin:4px 0 14px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:right}th:first-child,td:first-child{text-align:left}.selected{background:#f1f6fd;font-weight:500}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:760px){main{padding:25px 15px 50px}.snapshot,.architecture,.columns{grid-template-columns:1fr}.controls{align-items:stretch}.chart-wrap{overflow:hidden}.arch-step:not(:last-child)::after{display:none}}
</style>
</head>
<body>
<main>
<h1>Interactive wireless/DSP knowledge-neuron benchmark</h1>
<p class="lede">这个页面不是静态 markdown：选择任务、SNR、方法开关，图表、架构图、结果快照和解释会同步更新。最终性能使用任务指标，不把不同 neuron 的 surrogate loss 直接横向比较。</p>
<div class="stats">
  <div class="stat"><strong id="deviceName"></strong><span>训练设备</span></div>
  <div class="stat"><strong id="torchInfo"></strong><span>PyTorch / CUDA</span></div>
  <div class="stat"><strong id="runtime"></strong><span>完整 benchmark 时间</span></div>
  <div class="stat"><strong id="paramCount"></strong><span>各任务 NN 的可训练参数总量</span></div>
</div>
<div class="controls" aria-label="Benchmark controls">
  <label>任务<select id="taskSelect"></select></label>
  <label>SNR<select id="snrSelect"></select></label>
  <label class="method-toggle"><input type="checkbox" data-method="Model-based" checked><span class="swatch gray"></span>Model-based</label>
  <label class="method-toggle"><input type="checkbox" data-method="Plain NN" checked><span class="swatch blue"></span>Plain NN</label>
  <label class="method-toggle"><input type="checkbox" data-method="Knowledge-neuron NN" checked><span class="swatch orange"></span>Knowledge-neuron NN</label>
</div>
<h2 id="taskTitle"></h2>
<p id="taskDescription" class="lede"></p>
<div class="chart-wrap">
  <svg id="metricChart" viewBox="0 0 900 390" role="img" aria-labelledby="chartTitle chartDesc">
    <title id="chartTitle">Interactive benchmark metric chart</title>
    <desc id="chartDesc">Use the controls above to change the task and visible methods.</desc>
  </svg>
</div>
<div id="snapshot" class="snapshot" aria-live="polite"></div>
<div class="columns">
  <section><h2>当前 SNR 的结果</h2><div id="resultTable"></div></section>
  <section><h2>当前任务的 architecture</h2><div id="architecture" class="architecture"></div></section>
</div>
<h2>这个知识 neuron 到底改变了什么？</h2>
<div id="modelExplanation"></div>
<div id="whyExplanation" class="explain" aria-live="polite"></div>
<details class="details"><summary>为什么 knowledge neuron 很多时候不比 model-based 方法好很多？</summary>
<p><strong>第一，model-based 方法可能已经接近信息论上限。</strong>如果信道模型、噪声方差和导频结构都正确，MMSE、匹配滤波或 FFT 峰值已经利用了任务最关键的结构；神经网络不能从同一份输入中凭空创造额外信息。</p>
<p><strong>第二，knowledge 不是“真值”。</strong>本实验的 q 来自名义信道、插值信道或 FFT/MMSE 估计；真实信道失配、噪声估计错误、模型简化都会把先验变成 bias。硬注入可能限制 NN，尤其是在先验不可靠的 SNR 区域。</p>
<p><strong>第三，plain NN 可能自己学会同一个算子。</strong>数据足够、backbone 足够大时，CNN/Transformer/GRU 可以从样本中学出匹配滤波、频域平滑或检测规则，knowledge path 只剩下很小的边际收益。</p>
<p><strong>第四，真正有效的知识 neuron 通常要学会“信不信先验”。</strong>本版把固定 residual 改成了 learnable trust gate，并且尽量用 residual correction，而不是用 q 直接覆盖 NN 输出。这样 knowledge neuron 的作用是提供低方差起点，允许网络在失配时绕开它。</p>
</details>
<details class="details"><summary>为什么不拿不同 neuron 的 loss 直接比较？</summary>
<p>因为这里的 loss 只是优化 surrogate：BPSK/MIMO 用 BCE，OFDM/单音用 MSE；输出尺度、目标分布、residual parameterization 和正则化也不同。最终比较应看 BER、channel NMSE、frequency MAE 等任务指标。loss 曲线仍保存在 benchmark_history.csv，供检查收敛使用。</p>
</details>
<h2>复现</h2>
<pre>python multi_example_benchmark.py --device auto
python multi_example_benchmark.py --device cuda
python multi_example_benchmark.py --quick</pre>
<p class="lede">当前完整运行使用 CUDA；如果另一台电脑安装了适配驱动和 CUDA 版 PyTorch，<code>--device auto</code> 会自动选择 GPU。结果不需要 push 回 GitHub，本地生成即可。</p>
</main>
<script>
const DATA = __DATA_JSON__;
const colors = {"Model-based":"#737b84","Plain NN":"#3f6fa6","Knowledge-neuron NN":"#e07820"};
const taskSelect = document.getElementById("taskSelect");
const snrSelect = document.getElementById("snrSelect");
const svg = document.getElementById("metricChart");
const snapshot = document.getElementById("snapshot");
const table = document.getElementById("resultTable");
const architecture = document.getElementById("architecture");
const taskTitle = document.getElementById("taskTitle");
const taskDescription = document.getElementById("taskDescription");
const explanation = document.getElementById("modelExplanation");
const why = document.getElementById("whyExplanation");
const methods = ["Model-based","Plain NN","Knowledge-neuron NN"];
document.getElementById("deviceName").textContent = DATA.summary.device_name;
document.getElementById("torchInfo").textContent = DATA.summary.torch_version + " / " + DATA.summary.torch_cuda;
document.getElementById("runtime").textContent = DATA.summary.total_runtime_seconds.toFixed(1) + " s";
document.getElementById("paramCount").textContent = DATA.summary.total_trainable_params.toLocaleString();
DATA.taskOrder.forEach((task) => {
  const option = document.createElement("option");
  option.value = task; option.textContent = DATA.taskMeta[task].title; taskSelect.appendChild(option);
});
document.querySelectorAll("[data-method]").forEach((input) => input.addEventListener("change", draw));
taskSelect.addEventListener("change", () => { updateSNR(); draw(); });
snrSelect.addEventListener("change", draw);
function currentTask(){ return taskSelect.value || DATA.taskOrder[0]; }
function rowsFor(task){ return DATA.metrics.filter((row) => row.task === task); }
function updateSNR(){
  const values = [...new Set(rowsFor(currentTask()).map((row) => row.snr_db))].sort((a,b) => a-b);
  snrSelect.innerHTML = values.map((v) => '<option value="' + v + '">' + v + ' dB</option>').join("");
  snrSelect.value = values[Math.min(3, values.length-1)];
}
function el(name, attrs={}, text=""){
  const n = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([k,v]) => n.setAttribute(k, v));
  if(text) n.textContent = text; return n;
}
function draw(){
  const task = currentTask(), meta = DATA.taskMeta[task], rows = rowsFor(task);
  taskTitle.textContent = meta.title; taskDescription.textContent = meta.description;
  const visible = methods.filter((method) => document.querySelector("[data-method='" + method + "']").checked);
  const snr = Number(snrSelect.value);
  const metricValues = rows.map((row) => Number(row.value)).filter((v) => v > 0);
  const logY = task === "bpsk" || task === "ofdm";
  const W=900,H=390,M={l:78,r:20,t:42,b:55},pw=W-M.l-M.r,ph=H-M.t-M.b;
  svg.innerHTML = "";
  svg.appendChild(el("title",{id:"chartTitle"},meta.title + " " + meta.metric));
  svg.appendChild(el("desc",{id:"chartDesc"},"Lower values are better. Hover points for exact values."));
  const minX=Math.min(...rows.map((r)=>Number(r.snr_db))), maxX=Math.max(...rows.map((r)=>Number(r.snr_db)));
  const minV=Math.min(...metricValues), maxV=Math.max(...metricValues);
  const yLow=logY ? Math.pow(10,Math.log10(minV)-0.12) : Math.max(0,minV-(maxV-minV)*0.08);
  const yHigh=logY ? Math.pow(10,Math.log10(maxV)+0.12) : maxV+(maxV-minV)*0.08;
  const xScale=(x)=>M.l+(x-minX)/(maxX-minX)*pw;
  const yScale=(v)=>{ if(logY){const a=Math.log10(yLow),b=Math.log10(yHigh);return M.t+(b-Math.log10(v))/(b-a)*ph;} return M.t+(yHigh-v)/(yHigh-yLow)*ph; };
  [0,0.25,0.5,0.75,1].forEach((frac)=>{
    const y=M.t+frac*ph; svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:y,y2:y,class:"gridline"}));
    const value=logY ? Math.pow(10,Math.log10(yHigh)-frac*(Math.log10(yHigh)-Math.log10(yLow))) : yHigh-frac*(yHigh-yLow);
    svg.appendChild(el("text",{x:M.l-10,y:y+4,"text-anchor":"end",class:"axis-label"},value.toPrecision(3)));
  });
  rows.filter((r)=>r.method==="Model-based").forEach((r)=>{
    const x=xScale(Number(r.snr_db)); svg.appendChild(el("line",{x1:x,x2:x,y1:M.t+ph,y2:M.t+ph+5,class:"axis"}));
    svg.appendChild(el("text",{x:x,y:H-20,"text-anchor":"middle",class:"axis-label"},Number(r.snr_db)+" dB"));
  });
  svg.appendChild(el("line",{x1:M.l,x2:W-M.r,y1:M.t+ph,y2:M.t+ph,class:"axis"}));
  svg.appendChild(el("line",{x1:M.l,x2:M.l,y1:M.t,y2:M.t+ph,class:"axis"}));
  svg.appendChild(el("text",{x:W/2,y:H-2,"text-anchor":"middle",class:"axis-label"},"SNR (dB)"));
  svg.appendChild(el("text",{x:16,y:H/2,transform:"rotate(-90 16 "+H/2+")","text-anchor":"middle",class:"axis-label"},meta.metric+" (lower is better)"));
  visible.forEach((method,methodIndex)=>{
    const methodRows=rows.filter((r)=>r.method===method);
    const path=methodRows.map((r,i)=>(i?"L":"M")+xScale(Number(r.snr_db)).toFixed(1)+","+yScale(Number(r.value)).toFixed(1)).join(" ");
    svg.appendChild(el("path",{d:path,class:"series",stroke:colors[method],opacity:"0.95"}));
    methodRows.forEach((r)=>{
      const c=el("circle",{cx:xScale(Number(r.snr_db)),cy:yScale(Number(r.value)),r:6,class:"point",fill:colors[method]});
      c.addEventListener("mouseenter",()=>{c.setAttribute("r","9");});
      c.addEventListener("mouseleave",()=>{c.setAttribute("r","6");});
      c.addEventListener("focus",()=>{c.setAttribute("r","9");});
      c.addEventListener("blur",()=>{c.setAttribute("r","6");});
      c.setAttribute("tabindex","0"); c.setAttribute("aria-label",method+" at "+r.snr_db+" dB: "+Number(r.value).toPrecision(6));
      svg.appendChild(c);
    });
  });
  const legendX=M.l, legendY=18;
  visible.forEach((method,i)=>{
    const x=legendX+i*190; svg.appendChild(el("circle",{cx:x,cy:legendY-4,r:5,fill:colors[method]}));
    svg.appendChild(el("text",{x:x+10,y:legendY,class:"chart-legend"},method));
  });
  updateSnapshot(task,snr); updateTable(task,snr); updateArchitecture(meta); updateExplanation(meta);
}
function updateSnapshot(task,snr){
  const rows=rowsFor(task).filter((r)=>Number(r.snr_db)===snr);
  const values=rows.map((r)=>Number(r.value)), max=Math.max(...values);
  snapshot.innerHTML=rows.map((row)=>'<div class="bar-item"><div class="bar-label"><span>'+row.method+'</span><strong>'+Number(row.value).toPrecision(6)+'</strong></div><div class="bar-track"><div class="bar-fill '+(row.method==="Plain NN"?"blue":row.method==="Knowledge-neuron NN"?"orange":"gray")+'" style="width:'+Math.max(2,100*Number(row.value)/max)+'%"></div></div></div>').join("");
}
function updateTable(task,snr){
  const rows=rowsFor(task), meta=DATA.taskMeta[task];
  table.innerHTML='<table><thead><tr><th>Method</th><th>'+meta.metric+'</th><th>SNR</th></tr></thead><tbody>'+methods.map((method)=>{const row=rows.find((r)=>r.method===method&&Number(r.snr_db)===snr);return '<tr class="selected"><td>'+method+'</td><td>'+Number(row.value).toPrecision(8)+'</td><td>'+snr+' dB</td></tr>';}).join('')+'</tbody></table>';
}
function updateArchitecture(meta){
  architecture.innerHTML=meta.architecture.map((column)=>'<div class="arch-col"><h3>'+column.label+'</h3>'+column.steps.map((step)=>'<div class="arch-step">'+step+'</div>').join('')+'<div class="arch-note">'+column.note+'</div></div>').join('');
}
function updateExplanation(meta){
  explanation.innerHTML='<p><strong>Model-based：</strong>'+meta.model+'</p><p><strong>Plain NN architecture：</strong>'+meta.plain+'</p><p><strong>Knowledge-neuron change：</strong>'+meta.knowledge+'</p>';
  why.innerHTML='<strong>为什么这个任务里不一定大幅领先：</strong> '+meta.why;
}
updateSNR(); draw();
</script>
</body>
</html>"""
    html = html.replace("__DATA_JSON__", data_json)
    (output_dir / "benchmark_report.html").write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    epochs = 8 if args.quick else args.epochs
    train_size = 1000 if args.quick else args.train_size
    test_size = 800 if args.quick else args.test_size
    batch_size = args.batch_size
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")

    metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    total_params = 0
    per_task_runtime: dict[str, float] = {}

    task_started = time.perf_counter()
    bpsk_train = make_bpsk_dataset(
        train_size, rng.uniform(0.0, 12.0, train_size), rng
    )
    bpsk_models = [
        ("Plain NN", BPSKCNN(knowledge=False)),
        ("Knowledge-neuron NN", BPSKCNN(knowledge=True)),
    ]
    bpsk_train_x, bpsk_train_k, bpsk_train_y = map(torch.from_numpy, bpsk_train)
    bpsk_history = []
    for offset, (name, model) in enumerate(bpsk_models):
        total_params += parameter_count(model)
        bpsk_history.extend(
            train_model(
                "bpsk", model, name, bpsk_train_x, bpsk_train_k, bpsk_train_y,
                device, epochs, batch_size, args.seed + offset,
            )
        )
    for snr in args.snr_grid:
        x_np, k_np, y_np = make_bpsk_dataset(test_size, snr, rng)
        x, k, y = map(torch.from_numpy, (x_np, k_np, y_np))
        model_ber = float(((k >= 0).float() != y).float().mean().item())
        metric_rows.append({"task": "bpsk", "method": "Model-based", "snr_db": snr, "value": model_ber})
        for name, model in bpsk_models:
            value = predict_metric("bpsk", model, x, k, y, device)
            metric_rows.append({"task": "bpsk", "method": name, "snr_db": snr, "value": value})
    history_rows.extend(bpsk_history)
    per_task_runtime["bpsk"] = time.perf_counter() - task_started

    task_started = time.perf_counter()
    ofdm_train = make_ofdm_dataset(train_size, rng.uniform(0.0, 20.0, train_size), rng)
    ofdm_models = [
        ("Plain NN", OFDMTransformer(knowledge=False)),
        ("Knowledge-neuron NN", OFDMTransformer(knowledge=True)),
    ]
    ofdm_train_x = torch.from_numpy(ofdm_train[0])
    ofdm_train_k = torch.from_numpy(ofdm_train[1])
    ofdm_train_y = torch.from_numpy(ofdm_train[2])
    ofdm_history = []
    for offset, (name, model) in enumerate(ofdm_models):
        total_params += parameter_count(model)
        ofdm_history.extend(
            train_model(
                "ofdm", model, name, ofdm_train_x, ofdm_train_k, ofdm_train_y,
                device, epochs, batch_size, args.seed + 10 + offset,
            )
        )
    for snr in args.snr_grid:
        x_np, k_np, y_np, channel_np = make_ofdm_dataset(test_size, snr, rng)
        x, k, y = map(torch.from_numpy, (x_np, k_np, y_np))
        model_nmse = float(
            np.mean(np.sum((k_np - y_np) ** 2, axis=(1, 2)) / np.maximum(np.sum(y_np**2, axis=(1, 2)), 1e-8))
        )
        metric_rows.append({"task": "ofdm", "method": "Model-based", "snr_db": snr, "value": model_nmse})
        for name, model in ofdm_models:
            value = predict_metric("ofdm", model, x, k, y, device)
            metric_rows.append({"task": "ofdm", "method": name, "snr_db": snr, "value": value})
    history_rows.extend(ofdm_history)
    per_task_runtime["ofdm"] = time.perf_counter() - task_started

    task_started = time.perf_counter()
    tone_train = make_tone_dataset(train_size, rng.uniform(-2.0, 18.0, train_size), rng)
    tone_models = [
        ("Plain NN", FrequencyGRU(knowledge=False)),
        ("Knowledge-neuron NN", FrequencyGRU(knowledge=True)),
    ]
    tone_train_x = torch.from_numpy(tone_train[0])
    tone_train_k = torch.from_numpy(tone_train[1])
    tone_train_y = torch.from_numpy(tone_train[2])
    tone_history = []
    for offset, (name, model) in enumerate(tone_models):
        total_params += parameter_count(model)
        tone_history.extend(
            train_model(
                "tone", model, name, tone_train_x, tone_train_k, tone_train_y,
                device, epochs, batch_size, args.seed + 20 + offset,
            )
        )
    for snr in args.snr_grid:
        x_np, k_np, y_np, clean_np = make_tone_dataset(test_size, snr, rng)
        x, k, y = map(torch.from_numpy, (x_np, k_np, y_np))
        model_mae = float(np.mean(np.abs(k_np - y_np)))
        metric_rows.append({"task": "tone", "method": "Model-based", "snr_db": snr, "value": model_mae})
        for name, model in tone_models:
            value = predict_metric("tone", model, x, k, y, device)
            metric_rows.append({"task": "tone", "method": name, "snr_db": snr, "value": value})
    history_rows.extend(tone_history)
    per_task_runtime["tone"] = time.perf_counter() - task_started

    task_started = time.perf_counter()
    mimo_train = make_mimo_dataset(train_size, rng.uniform(0.0, 16.0, train_size), rng)
    mimo_models = [
        ("Plain NN", MIMOMLP(knowledge=False)),
        ("Knowledge-neuron NN", MIMOMLP(knowledge=True)),
    ]
    mimo_train_x = torch.from_numpy(mimo_train[0])
    mimo_train_k = torch.from_numpy(mimo_train[1])
    mimo_train_y = torch.from_numpy(mimo_train[2])
    mimo_history = []
    for offset, (name, model) in enumerate(mimo_models):
        total_params += parameter_count(model)
        mimo_history.extend(
            train_model(
                "mimo", model, name, mimo_train_x, mimo_train_k, mimo_train_y,
                device, epochs, batch_size, args.seed + 30 + offset,
            )
        )
    for snr in args.snr_grid:
        x_np, k_np, y_np = make_mimo_dataset(test_size, snr, rng)
        x, k, y = map(torch.from_numpy, (x_np, k_np, y_np))
        model_ber = float(((k >= 0).float() != y).float().mean().item())
        metric_rows.append({"task": "mimo", "method": "Model-based", "snr_db": snr, "value": model_ber})
        for name, model in mimo_models:
            value = predict_metric("mimo", model, x, k, y, device)
            metric_rows.append({"task": "mimo", "method": name, "snr_db": snr, "value": value})
    history_rows.extend(mimo_history)
    per_task_runtime["mimo"] = time.perf_counter() - task_started

    elapsed = time.perf_counter() - started
    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda or "none",
        "epochs": epochs,
        "train_size_per_task": train_size,
        "test_size_per_snr": test_size,
        "batch_size": batch_size,
        "snr_grid": args.snr_grid,
        "seed": args.seed,
        "total_trainable_params": total_params,
        "per_task_runtime_seconds": per_task_runtime,
        "total_runtime_seconds": elapsed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_csv(output_dir / "benchmark_metrics.csv", metric_rows, ["task", "method", "snr_db", "value"])
    write_csv(output_dir / "benchmark_history.csv", history_rows, ["task", "model", "epoch", "loss"])
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_benchmark(output_dir, metric_rows)
    build_interactive_report(output_dir, summary, metric_rows, history_rows)
    print(f"Saved benchmark report to {(output_dir / 'benchmark_report.html').resolve()}")
    print(f"Total runtime: {elapsed:.2f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--snr-grid",
        type=float,
        nargs="+",
        default=[-4.0, 0.0, 4.0, 8.0, 12.0, 16.0],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

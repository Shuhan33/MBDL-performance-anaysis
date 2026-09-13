"""Compare standard ReLU and knowledge-informed neurons on wireless BPSK detection."""

from __future__ import annotations

import argparse
import csv
import json
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


ARCHITECTURE = [256, 128, 64, 4, 1]
SEED = 7
N_SAMPLES = 128
TARGET_INDEX = 64
CHANNEL = np.array(
    [0.15 + 0.05j, 0.38 - 0.12j, 0.88 + 0.00j, 0.42 + 0.11j, 0.18 - 0.04j],
    dtype=np.complex64,
)
CHANNEL = CHANNEL / np.sqrt(np.mean(np.abs(CHANNEL) ** 2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(
    n: int,
    snr_db: float | np.ndarray,
    rng: np.random.Generator,
    channel_jitter: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw I/Q features, a matched-filter prior q, and BPSK labels."""
    symbols = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, N_SAMPLES))

    # The receiver knows a nominal channel, while the true channel varies a little.
    perturbation = (
        rng.normal(size=(n, CHANNEL.size))
        + 1j * rng.normal(size=(n, CHANNEL.size))
    ).astype(np.complex64)
    h_true = CHANNEL[None, :] * (1.0 + channel_jitter * perturbation / np.sqrt(2.0))
    h_true = h_true * (
        np.sqrt(np.mean(np.abs(CHANNEL) ** 2))
        / np.sqrt(np.mean(np.abs(h_true) ** 2, axis=1, keepdims=True))
    )

    received = np.zeros((n, N_SAMPLES), dtype=np.complex64)
    for tap, coefficient in enumerate(h_true.T):
        received[:, tap:] += coefficient[:, None] * symbols[:, : N_SAMPLES - tap]

    snr_values = np.broadcast_to(np.asarray(snr_db, dtype=np.float32), (n,))
    snr_linear = np.power(10.0, snr_values / 10.0)
    signal_power = np.mean(np.abs(received) ** 2, axis=1)
    noise_power = signal_power / snr_linear
    noise = (
        rng.normal(size=received.shape) + 1j * rng.normal(size=received.shape)
    ) * np.sqrt(noise_power[:, None] / 2.0)
    received = received + noise.astype(np.complex64)

    # q is the known-physics feature: a nominal-channel matched-filter score.
    window = received[:, TARGET_INDEX : TARGET_INDEX + CHANNEL.size]
    q = np.real(np.sum(np.conj(CHANNEL[None, :]) * window, axis=1)) / np.sum(
        np.abs(CHANNEL) ** 2
    )

    features = np.concatenate([received.real, received.imag], axis=1).astype(np.float32)
    labels = (symbols[:, TARGET_INDEX] > 0).astype(np.float32).reshape(-1, 1)
    return features, q.astype(np.float32).reshape(-1, 1), labels


class StandardDNN(nn.Module):
    """256-128-64-4-1 MLP with ordinary ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(ARCHITECTURE[i], ARCHITECTURE[i + 1])
                for i in range(len(ARCHITECTURE) - 1)
            ]
        )

    def forward(self, x: Tensor, q: Tensor | None = None) -> Tensor:
        del q
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class KnowledgeReLU(nn.Module):
    """ReLU plus a bounded physics residual, with no trainable parameters."""

    def __init__(self, signed_path: float = 0.25) -> None:
        super().__init__()
        self.signed_path = signed_path

    def forward(self, z: Tensor, q: Tensor) -> Tensor:
        q = torch.tanh(q).view(-1, 1)
        return F.relu(z) + self.signed_path * q


class KnowledgeDNN(nn.Module):
    """The same trainable layers, with KnowledgeReLU in hidden layers."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(ARCHITECTURE[i], ARCHITECTURE[i + 1])
                for i in range(len(ARCHITECTURE) - 1)
            ]
        )
        self.knowledge_relu = KnowledgeReLU()

    def forward(self, x: Tensor, q: Tensor | None = None) -> Tensor:
        if q is None:
            raise ValueError("KnowledgeDNN requires the matched-filter score q")
        for layer in self.layers[:-1]:
            x = self.knowledge_relu(layer(x), q)
        return self.layers[-1](x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def evaluate(model: nn.Module, x: Tensor, q: Tensor, y: Tensor) -> dict[str, float]:
    model.eval()
    logits = model(x, q)
    predictions = (logits >= 0).float()
    accuracy = float((predictions == y).float().mean().item())
    return {"accuracy": accuracy, "ber": 1.0 - accuracy}


def train_model(
    model: nn.Module,
    name: str,
    x: Tensor,
    q: Tensor,
    y: Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    model.to(device)
    loader = DataLoader(
        TensorDataset(x, q, y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    x_eval, q_eval, y_eval = x.to(device), q.to(device), y.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for xb, qb, yb in loader:
            xb, qb, yb = xb.to(device), qb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, qb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total_items += xb.shape[0]

        metrics = evaluate(model, x_eval, q_eval, y_eval)
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "loss": total_loss / max(total_items, 1),
                "train_accuracy": metrics["accuracy"],
                "train_ber": metrics["ber"],
            }
        )
        if epoch == 1 or epoch == epochs or epoch % max(epochs // 5, 1) == 0:
            print(
                f"{name:22s} epoch {epoch:3d}/{epochs} "
                f"loss={history[-1]['loss']:.4f} "
                f"train_BER={metrics['ber']:.4f}"
            )
    return history


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    output_dir: Path,
    history: list[dict[str, float]],
    metrics: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {"Standard ReLU DNN": "#4c78a8", "Knowledge-informed DNN": "#f58518"}
    for model_name, color in colors.items():
        model_history = [row for row in history if row["model"] == model_name]
        axes[0].plot(
            [row["epoch"] for row in model_history],
            [row["loss"] for row in model_history],
            label=model_name,
            color=color,
            linewidth=2,
        )
        model_metrics = [row for row in metrics if row["model"] == model_name]
        axes[1].plot(
            [row["snr_db"] for row in model_metrics],
            [row["ber"] for row in model_metrics],
            marker="o",
            label=model_name,
            color=color,
            linewidth=2,
        )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss (lower is better)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].set_title("BPSK detection BER")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("BER (log scale, lower is better)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "training_and_ber.png", dpi=170)
    plt.close(fig)


def build_report(
    output_dir: Path,
    config: dict[str, object],
    summary: dict[str, object],
    metrics: list[dict[str, object]],
) -> None:
    model_names = ["Standard ReLU DNN", "Knowledge-informed DNN"]
    snr_values = sorted(
        {float(row["snr_db"]) for row in metrics if row["model"] in model_names}
    )
    table_rows = []
    for snr in snr_values:
        cells = [f"<td>{snr:g}</td>"]
        for model_name in model_names:
            item = next(
                row
                for row in metrics
                if row["model"] == model_name and float(row["snr_db"]) == snr
            )
            cells.append(
                f"<td>{float(item['ber']):.4f}</td><td>{100*float(item['accuracy']):.2f}%</td>"
            )
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge-informed neuron wireless experiment</title>
<style>
:root {{ color-scheme:light; --ink:#18202a; --muted:#5d6875; --line:#d9dee5; --accent:#1f6feb; --soft:#f5f7fa; }}
body {{ margin:0; color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
main {{ max-width:1000px; margin:0 auto; padding:42px 24px 72px; }}
h1 {{ font-size:30px; line-height:1.2; margin:0 0 10px; }} h2 {{ margin-top:36px; font-size:21px; }} h3 {{ margin-top:24px; font-size:17px; }}
.lede {{ color:var(--muted); font-size:16px; max-width:800px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:24px 0; }}
.stat {{ background:var(--soft); border:1px solid var(--line); padding:14px 16px; }} .stat strong {{ display:block; font-size:22px; }} .stat span {{ color:var(--muted); font-size:13px; }}
img {{ max-width:100%; border:1px solid var(--line); }} table {{ border-collapse:collapse; width:100%; font-size:14px; }} th,td {{ border-bottom:1px solid var(--line); text-align:right; padding:8px; }} th:first-child,td:first-child {{ text-align:left; }}
code,pre {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }} code {{ background:var(--soft); padding:2px 4px; }} pre {{ background:#111827; color:#e5e7eb; overflow:auto; padding:14px; }}
.note {{ border-left:3px solid var(--accent); padding:8px 14px; background:#f7fbff; }} a {{ color:var(--accent); }} small {{ color:var(--muted); }}
</style></head><body><main>
<h1>Knowledge-informed neuron：无线多径 BPSK 检测对比</h1>
<p class="lede">同一套 256–128–64–4–1 全连接网络，在相同数据、随机种子和优化器下，对比普通 ReLU 与带有匹配滤波知识的类神经元。当前主结果是 data-limited 条件：训练集仅 {summary['train_size']} 个样本。</p>
<div class="grid">
<div class="stat"><strong>{summary['device']}</strong><span>运行设备</span></div>
<div class="stat"><strong>{summary['architecture']}</strong><span>网络宽度</span></div>
<div class="stat"><strong>{summary['epochs']}</strong><span>训练轮数</span></div>
<div class="stat"><strong>{summary['standard_params']}</strong><span>每个模型的可训练参数</span></div>
</div>
<h2>结果</h2>
<p>BER 越低越好；accuracy 是中心 BPSK 符号的判决准确率。<small>测试集的真实信道相对名义信道有 10% 随机扰动，因此先验不是完美答案。</small></p>
<img src="training_and_ber.png" alt="训练损失与不同 SNR 下的 BER 曲线">
<table><thead><tr><th>SNR (dB)</th><th colspan="2">Standard ReLU DNN</th><th colspan="2">Knowledge-informed DNN</th></tr>
<tr><th></th><th>BER</th><th>Accuracy</th><th>BER</th><th>Accuracy</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p><small>完整数值：<a href="metrics_by_snr.csv">metrics_by_snr.csv</a>；训练曲线：<a href="training_history.csv">training_history.csv</a>。</small></p>
<h2>实验到底改变了什么</h2>
<p>每个样本是长度为 128 的复数接收序列 <code>y</code>，输入网络前拆成 128 个 I 分量和 128 个 Q 分量，所以输入维度是 256。序列由 BPSK 符号经过 5-tap 复数多径信道并加入 AWGN 得到；网络要判断中心位置的符号。</p>
<p>普通模型的隐藏层使用 <code>ReLU(z)=max(0,z)</code>。知识模型使用固定的、无额外可训练参数的：</p>
<pre>KReLU(z,q) = ReLU(z + 0.35*tanh(q)) + 0.08*tanh(q)</pre>
<p>其中 <code>q</code> 是用名义信道冲激响应做的可微匹配滤波分数。它把无线通信中“已知信道结构可以先做匹配/合并”的知识注入每个隐藏神经元；权重层、宽度、损失函数和训练数据都保持不变。</p>
<div class="note"><strong>解释边界：</strong>这个结果证明的是“在这个合成多径检测任务和这个先验注入方式下，知识偏置是否有帮助”，不是普遍证明所有 knowledge neuron 都优于 ReLU。下一步可做信道失配、未知信道、不同 tap 数、不同先验强度的消融实验。</div>
<h2>如何运行</h2>
<h3>直接用 Python 3.12</h3>
<pre>python -m pip install -r requirements.txt
python train_experiment.py --device auto --epochs 30</pre>
<p>若已安装 CUDA 版 PyTorch，<code>--device auto</code> 会自动使用 GPU；也可以显式写 <code>--device cuda</code>：</p>
<pre>python -c "import torch; print(torch.__version__, torch.cuda.is_available())"</pre>
<p>快速试跑：<code>python train_experiment.py --quick</code>。所有输出写入 <code>results/</code>，不会删除原有文件。</p>
<h3>Docker（CPU，可复现）</h3>
<pre>docker build -t knowledge-neuron-wireless .
docker run --rm -v "${{PWD}}/results:/work/results" knowledge-neuron-wireless</pre>
<p>这个 Dockerfile 使用 Python 3.12 和 CPU 版 PyTorch。GPU Docker 需要宿主机安装 NVIDIA Container Toolkit，并在镜像中改用与你驱动匹配的 CUDA 版 PyTorch；直接在已有 CUDA Python 环境运行通常更省事。</p>
<h3>可复现参数</h3>
<pre>{config_json}</pre>
<p>本项目不会运行 MATLAB；如果之后希望在有 license 的远程机器上做 MATLAB 复现，可以按同一组数据公式和 <code>metrics_by_snr.csv</code> 对照。Python 版本是完整自动实验入口。</p>
<h2>参考</h2>
<ul><li><a href="https://arxiv.org/abs/1702.00832">O'Shea &amp; Hoydis, An Introduction to Deep Learning for the Physical Layer</a>：将深度学习用于物理层和 expert/domain knowledge 的背景。</li>
<li><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html">PyTorch ReLU documentation</a>：普通 ReLU 定义。</li>
<li><a href="https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html">PyTorch CUDA availability documentation</a>：本脚本自动选择 GPU 的依据。</li></ul>
<p><small>生成时间：{summary['generated_at']}；耗时：{summary['elapsed_seconds']:.1f} 秒。</small></p>
</main></body></html>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size-per-snr", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true", help="Small smoke test")
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 8)
        args.train_size = min(args.train_size, 4000)
        args.test_size_per_snr = min(args.test_size_per_snr, 800)

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    print(f"Using device: {device}")
    print(f"Architecture: {'-'.join(map(str, ARCHITECTURE))}")

    x_np, q_np, y_np = make_dataset(
        args.train_size, rng.uniform(0.0, 12.0, size=args.train_size), rng
    )
    x_train, q_train, y_train = map(torch.from_numpy, (x_np, q_np, y_np))
    torch.manual_seed(args.seed + 100)
    standard = StandardDNN()
    torch.manual_seed(args.seed + 100)
    knowledge = KnowledgeDNN()
    assert parameter_count(standard) == parameter_count(knowledge)

    history = []
    history.extend(
        train_model(
            standard, "Standard ReLU DNN", x_train, q_train, y_train, device,
            args.epochs, args.batch_size, args.learning_rate, args.seed + 1,
        )
    )
    history.extend(
        train_model(
            knowledge, "Knowledge-informed DNN", x_train, q_train, y_train, device,
            args.epochs, args.batch_size, args.learning_rate, args.seed + 2,
        )
    )

    snr_grid = [-4.0, 0.0, 4.0, 8.0, 12.0, 16.0]
    metrics: list[dict[str, object]] = []
    for snr_db in snr_grid:
        x_test_np, q_test_np, y_test_np = make_dataset(args.test_size_per_snr, snr_db, rng)
        x_test = torch.from_numpy(x_test_np).to(device)
        q_test = torch.from_numpy(q_test_np).to(device)
        y_test = torch.from_numpy(y_test_np).to(device)
        for model_name, model in [("Standard ReLU DNN", standard), ("Knowledge-informed DNN", knowledge)]:
            result = evaluate(model, x_test, q_test, y_test)
            metrics.append({"model": model_name, "snr_db": snr_db, **result})
        prior_predictions = (q_test >= 0).float()
        prior_accuracy = float((prior_predictions == y_test).float().mean().item())
        metrics.append({
            "model": "Matched-filter prior only", "snr_db": snr_db,
            "ber": 1.0 - prior_accuracy, "accuracy": prior_accuracy,
        })

    save_csv(output_dir / "training_history.csv", history,
             ["model", "epoch", "loss", "train_accuracy", "train_ber"])
    save_csv(output_dir / "metrics_by_snr.csv", metrics,
             ["model", "snr_db", "ber", "accuracy"])
    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "device": str(device), "architecture": "-".join(map(str, ARCHITECTURE)),
        "epochs": args.epochs, "train_size": args.train_size,
        "test_size_per_snr": args.test_size_per_snr, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "standard_params": parameter_count(standard), "knowledge_params": parameter_count(knowledge),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": elapsed,
    }
    config = {
        "architecture": ARCHITECTURE,
        "nominal_channel_IQ": [[float(v.real), float(v.imag)] for v in CHANNEL],
        "target_index": TARGET_INDEX, "training_snr_db": "Uniform(0, 12)",
        "test_snr_db": snr_grid, "channel_jitter": 0.10,
        "knowledge_neuron": "ReLU(z) + 0.25*tanh(q)",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(output_dir, history, metrics)
    build_report(output_dir, config, summary, metrics)
    print(f"Saved results to: {output_dir.resolve()}")
    print(f"Report: {(output_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    main()

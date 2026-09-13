# Knowledge-Neuron Robustness Benchmarks

This subproject evaluates whether a smaller knowledge-neuron network can match or exceed a much larger paper-inspired neural baseline when the test distribution changes. The comparison is deliberately broader than a single SNR curve.

## Scenarios

### 1. MIMO detection under distribution shift

The large baseline is a DetNet-style unfolded detector with 12 stages and 128 hidden units per stage. It is inspired by:

- N. Samuel, T. Diskin, and A. Wiesel, “Deep MIMO Detection,” 2017, [arXiv:1706.01151](https://arxiv.org/abs/1706.01151).

The knowledge-neuron variants use the same physical residual/gradient structure but have fewer stages and hidden units. Each stage receives an MMSE estimate as a soft prior and applies a bounded knowledge correction with a learnable trust mechanism.

Training uses iid Rayleigh 8x8 real MIMO channels. Evaluation includes:

- iid Rayleigh channels;
- strongly correlated Rayleigh channels;
- Rician channels;
- wrong noise-variance information;
- impulsive noise;
- low- and high-SNR shifts.

### 2. OFDM reception under waveform impairment shift

The large baseline is a fully convolutional residual receiver inspired by DeepRx:

- M. Honkala, D. Korpi, and J. M. J. Huttunen, “DeepRx: Fully Convolutional Deep Learning Receiver,” 2020, [arXiv:2005.01494](https://arxiv.org/abs/2005.01494).

The knowledge-neuron variants are compact residual CNNs. They use the classical pilot-based equalized symbol estimate `q` as a soft prior, predict a correction, and learn a spatially varying trust mask instead of blindly replacing the neural output.

Training uses nominal short-delay 64-subcarrier QPSK OFDM. Evaluation includes:

- nominal channels;
- longer delay spread;
- unseen carrier-frequency offset;
- impulsive noise;
- low- and high-SNR shifts.

## What is measured?

For every model and condition, the benchmark records:

- BER;
- MSE and EVM;
- trainable parameter count;
- wall-clock training time;
- final training loss as a diagnostic only;
- robustness gap, defined as the worst out-of-distribution metric divided by the in-distribution metric minus one;
- sample-efficiency curves for the large baseline and the medium knowledge-neuron model.

The benchmark intentionally does not assume that knowledge neurons must win at every SNR. A correct model-based method can already be very strong, while the main value of a knowledge path may be lower sample demand, lower parameter count, faster training, or smaller degradation under mismatch.

## Run

From this directory:

```powershell
python -m pip install -r ../knowledge_informed_wireless_neuron/requirements.txt
python run_robustness_benchmark.py --device auto
```

To force CUDA:

```powershell
python run_robustness_benchmark.py --device cuda
```

For a short smoke test:

```powershell
python run_robustness_benchmark.py --quick
```

The default run creates `results/robustness_report.html`, `results/robustness_metrics.csv`, `results/sample_efficiency.csv`, `results/robustness_summary.json`, and `results/robustness_summary.png`.

The report is self-contained and interactive: select a scenario, distribution-shift condition, and metric to update the chart and table.

## References and background

The broader physical-layer deep-learning context is discussed in:

- T. J. O'Shea and J. Hoydis, “An Introduction to Deep Learning for the Physical Layer,” [arXiv:1702.00832](https://arxiv.org/abs/1702.00832).
- Z. Qin, H. Ye, G. Y. Li, and B.-H. F. Juang, “Deep Learning in Physical Layer Communications,” [arXiv:1807.11713](https://arxiv.org/abs/1807.11713).
- H. He, C.-K. Wen, S. Jin, and G. Y. Li, “Model-Driven Deep Learning for MIMO Detection,” IEEE Transactions on Signal Processing, 2020, [DOI:10.1109/TSP.2020.2976585](https://doi.org/10.1109/TSP.2020.2976585).
- A. Vaswani et al., “Attention Is All You Need,” [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), for the Transformer family used in the earlier benchmark.
- PyTorch [local installation](https://pytorch.org/get-started/locally/) and [CUDA availability](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html) documentation.

The knowledge-neuron modules in this repository are experimental implementations for controlled parameter-efficiency and robustness comparisons; they are not presented as canonical reproductions of any single paper.

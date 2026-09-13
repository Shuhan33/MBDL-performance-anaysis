# MBDL Performance Analysis

This repository is a growing collection of model-based deep learning (MBDL) simulations for wireless communications and digital signal processing. The goal is to compare classical signal-processing methods, ordinary neural networks, and neural networks that explicitly incorporate domain knowledge.

The repository is intentionally organized so that future MBDL experiments can be added as independent subprojects without changing the existing experiments.

## Current experiment

The first subproject is [`knowledge_informed_wireless_neuron`](knowledge_informed_wireless_neuron/). It studies whether replacing ordinary nonlinearities with knowledge-informed neuron variants improves performance on wireless/DSP tasks.

The benchmark compares three results for every task:

1. **Model-based:** a conventional signal-processing or communication-theory reference;
2. **Plain NN:** a trainable neural network without an explicit knowledge path;
3. **Knowledge-neuron NN:** the same general backbone with a task-specific prior, residual correction, and learnable trust when appropriate.

The current benchmark contains four examples:

- **Multipath BPSK detection:** model-based matched filtering versus a larger 1-D CNN;
- **OFDM channel estimation:** interpolation versus a Transformer encoder;
- **Single-tone frequency estimation:** FFT peak estimation versus a GRU;
- **2x2 MIMO QPSK detection:** linear MMSE detection versus a wider MLP.

The HTML report is interactive. It supports task selection, SNR selection, method toggles, metric curves, current-SNR tables, architecture diagrams, and explanations of what the knowledge neuron changes:

[`benchmark_report.html`](knowledge_informed_wireless_neuron/benchmark_results/benchmark_report.html)

The stored CSV and JSON files in `benchmark_results/` contain the numerical results and run metadata. These results are one reproducible benchmark run, not a universal claim that knowledge-informed neurons must outperform either a plain NN or a model-based method.

## Why compare task metrics instead of raw loss?

The experiments use BER, channel NMSE, and frequency-estimation MAE as final metrics. A loss value is a training surrogate and depends on the output parameterization, scaling, residual formulation, and optimization objective. Therefore, loss values from different neuron designs should not be interpreted as directly comparable evidence of task performance.

Knowledge can also fail to produce a large gain. A model-based method may already be close to optimal when its assumptions are correct; a prior may be biased under channel or noise mismatch; and a sufficiently large plain NN may learn a similar operator from data. The intended research question is broader than “does the knowledge neuron always win?”: we will also study sample efficiency, robustness to model mismatch, convergence, calibration, and out-of-distribution performance.

## Run the current experiment

The code supports Python 3.12 and automatically selects CUDA when it is available.

```powershell
cd knowledge_informed_wireless_neuron
python -m pip install -r requirements.txt
python multi_example_benchmark.py --device auto
```

To force CUDA:

```powershell
python multi_example_benchmark.py --device cuda
```

For a short smoke test:

```powershell
python multi_example_benchmark.py --quick
```

The benchmark writes the report and numerical results to `benchmark_results/`. The current full configuration uses four tasks, 4,000 training samples per task, 15 epochs, and six SNR points. On the development machine it took less than one minute on an NVIDIA RTX 2060. A newer laptop GPU may be faster, but the small models mean that the speedup will not necessarily scale linearly with peak GPU throughput.

For a CUDA installation on another computer, use the official PyTorch selector for the target operating system, Python version, driver, and CUDA wheel rather than copying a wheel from another machine:

- <https://pytorch.org/get-started/locally/>

The other computer only needs to pull this repository and run the benchmark. Results are generated locally; no push-back to GitHub is required.

## Repository structure

```text
mbdl_performance_analysis/
├── README.md
└── knowledge_informed_wireless_neuron/
    ├── multi_example_benchmark.py
    ├── benchmark_results/
    │   ├── benchmark_report.html
    │   ├── benchmark_metrics.csv
    │   ├── benchmark_metrics.png
    │   └── benchmark_summary.json
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
```

Future subprojects can follow the same pattern, for example:

```text
mbdl_performance_analysis/
├── knowledge_informed_wireless_neuron/
├── learned_receiver_comparison/
├── physics_informed_ofdm/
└── mimo_channel_mismatch_study/
```

## References

The current experiment is inspired by the following physical-layer deep-learning, wireless, sequence-modeling, and implementation references:

1. T. J. O'Shea and J. Hoydis, “An Introduction to Deep Learning for the Physical Layer,” 2017. [arXiv:1702.00832](https://arxiv.org/abs/1702.00832).
2. N. Samuel, T. Diskin, and A. Wiesel, “Deep MIMO Detection,” 2017. [arXiv:1706.01151](https://arxiv.org/abs/1706.01151).
3. A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
4. K. Cho et al., “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP 2014. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078).
5. A. Goldsmith, *Wireless Communications*, Cambridge University Press, 2005. This is background for fading channels, detection, and link-level metrics.
6. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2009. This is background for sampled signals, FFT-based estimation, and filtering.
7. PyTorch documentation for [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [CUDA availability](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html), and [local installation](https://pytorch.org/get-started/locally/).

The knowledge-neuron variants in this repository are experimental implementations for controlled comparisons. They should not be treated as a canonical implementation of any single paper.

## License and research status

This repository is research code intended for reproducible experiments and future MBDL studies. Add a project-specific license and citation file before redistributing the repository as a public research artifact.

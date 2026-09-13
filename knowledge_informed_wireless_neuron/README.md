# Knowledge-informed neuron for wireless/DSP experiments

当前推荐入口是 `multi_example_benchmark.py`。它在四个不同任务上比较三类结果：

1. Model-based：传统信号处理/通信方法；
2. Plain NN：普通神经网络；
3. Knowledge-neuron NN：相同或等价的可训练 backbone，加上任务知识残差/先验。

四个例子分别使用不同架构：

- 多径 BPSK 检测：1D CNN；
- OFDM 信道估计：小型 Transformer encoder；
- DSP 单音频率估计：GRU。
- 2×2 MIMO QPSK 检测：宽 MLP，并与线性 MMSE 检测器比较。

运行后的主报告是 `benchmark_results/benchmark_report.html`。这是一个交互式页面，不是静态 markdown：可以切换任务、SNR、显示的方法，并同步查看曲线、当前 SNR 快照、模型架构、knowledge neuron 的改动和展开式解释。

## 四例 benchmark

```powershell
python multi_example_benchmark.py --device auto
```

强制 CUDA：

```powershell
python multi_example_benchmark.py --device cuda
```

快速 smoke test：

```powershell
python multi_example_benchmark.py --quick
```

本机 RTX 2060 的完整配置（每个任务 4,000 个训练样本、15 epochs、6 个 SNR 点）实测总耗时约 39 秒，因此不需要额外打包。`benchmark_results/benchmark_summary.json` 会记录实际设备、PyTorch/CUDA 版本和时间。

最终横向比较使用 BER、信道 NMSE、频率 MAE 等任务指标。不同 neuron 的 loss 只作为训练诊断，不把 loss 数值直接当作跨方法性能结论。

### 远程 RTX PRO 500 Blackwell Laptop

另一台电脑只需要把项目 pull 下来后，在目标文件夹运行同一条命令；不需要向 GitHub push 结果：

```powershell
python -m pip install numpy matplotlib
```

然后按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为 Windows、Python、CUDA 版本安装 CUDA wheel，再运行：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python multi_example_benchmark.py --device auto
```

不要直接把当前机器的 `cu126` 安装命令硬编码到 Blackwell 电脑；以那台机器的驱动和 PyTorch 官方选择器为准。

当前 benchmark 在 RTX 2060 上只有约 39 秒，Blackwell 机器未必能按比例加速，因为模型和数据都很小，kernel launch 与数据搬运开销占比可能较高。增大样本数、Transformer 层数或 Monte-Carlo 重复次数后，GPU 差异才更容易体现；因此目前不建议为了这组实验额外折腾 Docker。

---

## 历史的单任务 baseline（保留用于对照）

下面的 `train_experiment.py` 是最初的单任务版本；当前推荐使用上面的四例 benchmark。这个旧版本用来比较：

- 普通 ReLU DNN：`256-128-64-4-1`
- Knowledge-informed DNN：完全相同的线性层和宽度，只把隐藏层激活替换为带匹配滤波先验的 `KReLU`

任务是从 128 个复数接收样本的 I/Q 分量中检测中心 BPSK 符号。数据经过 5-tap 复数多径信道和 AWGN 生成；真实信道相对名义信道有 10% 的随机扰动。知识分数 `q` 使用名义信道冲激响应做匹配滤波，因此先验有帮助但不是完美答案。

## 直接运行

```powershell
python -m pip install -r requirements.txt
python train_experiment.py --device auto --epochs 30
```

也可以运行一键 PowerShell 脚本：

```powershell
.\run_experiment.ps1
```

快速 smoke test：

```powershell
python train_experiment.py --quick
```

运行结束后打开 `results/report.html`。默认是 data-limited 条件（1,000 个训练样本），报告包含训练曲线、不同 SNR 下的 BER、完整参数和复现说明。

如果希望看 data-rich 对照，把训练集扩大到 18,000 个样本：

```powershell
python train_experiment.py --train-size 18000 --epochs 30 --device auto
```

## CUDA

脚本默认使用 `--device auto`：如果 `torch.cuda.is_available()` 为真就使用 GPU，否则使用 CPU。已经装好 CUDA 版 PyTorch 时无需改代码：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python train_experiment.py --device cuda --epochs 30
```

如果还没有 CUDA 版 PyTorch，请按照 PyTorch 官网针对你的驱动和 CUDA 版本选择对应安装命令；不要盲目复用其他机器的 CUDA wheel。

## Docker

Dockerfile 提供 CPU 版、固定 Python 3.12 的可复现环境：

```powershell
docker build -t knowledge-neuron-wireless .
docker run --rm -v "${PWD}/results:/work/results" knowledge-neuron-wireless
```

它不会删除工作区文件，只会把新的结果写入 `results/`。GPU Docker 需要宿主机的 NVIDIA Container Toolkit，并且需要改用与你驱动匹配的 CUDA 版 PyTorch 镜像；在已有 CUDA Python 环境中直接运行通常更简单。

## 公平性边界

两个模型的可训练参数数量相同：线性层都是 `256-128-64-4-1`。KnowledgeReLU 的 `0.25` 是固定常数，不增加可训练参数。两者共享同一批训练数据、损失、优化器和随机初始化。知识模型额外使用的是已知的名义信道匹配滤波分数，这是对普通 ReLU 的有意 inductive bias；因此结果应理解为一个可复现实验，而不是对所有知识神经元设计的普适结论。data-rich 条件下普通 ReLU 可能追平甚至胜出，这正是值得继续研究的边界。

## 参考

- O'Shea & Hoydis, [An Introduction to Deep Learning for the Physical Layer](https://arxiv.org/abs/1702.00832)
- [PyTorch ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- [PyTorch CUDA availability](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)

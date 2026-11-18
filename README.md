# MultiScaleScannerNet: Hierarchical Global, Group, and Running 1→1 Micro-Scanners

**MultiScaleScannerNet** is a hierarchical PyTorch architecture that inspects hidden representations simultaneously across **five distinct structural scales** before fusing them through a residual arbiter and an exit mini-network.

---

## 💡 Core Architectural Idea

Each 100-unit layer partitions neurons into $G$ groups of size $S$ ($G \times S = 100$) and processes the layer through five parallel observational streams:

1. **Global Aggregator (`100 → 16`):** Reads the entire 100-unit layer to capture macro-level system state.
2. **Global Running Scanner (`1 → 1` across all 100 units):** A single shared micro-network sweeps across every neuron in the layer, acting as a unified, learnable non-linear activation curve.
3. **Group Aggregators ($G$ independent nets, `S → 1`):** Each group has its own dedicated sub-network summarizing its $S$ neurons into a single local verdict.
4. **Group Running Scanners ($G$ independent nets, `1 → 1` inside each group):** Each group possesses a dedicated 1→1 micro-network sweeping across its $S$ neurons, providing cluster-specific non-linear filtering without quadratic parameter explosion.
5. **Cross-Group Observers ($S$ strided nets, `G → 1`):** Sample one neuron from each distant group (strided routing), capturing long-range dependencies across non-adjacent feature clusters.

All streams are concatenated into a **Layer Arbiter** with a residual connection, stacked across multiple layers, and passed into an **Exit Mini-Network**.

---

## 📊 Experimental Results (Enveda CASMI 2026 Mass Spectra)

### 1. Ablation Study (1,000 Strictly Unseen Molecules `inchikey14`)
Trained via symmetric contrastive learning on **80,000 paired MS2 spectra** from the 2.54M spectrum CASMI 2026 dataset:

| Configuration | Trainable Params | Unseen Top-1 Acc ↑ | Unseen Top-25 Acc ↑ |
| :--- | :---: | :---: | :---: |
| **1. Base (`10×10`, No Cross-Groups)** | 76,414 | 56.40% | **97.20%** |
| **2. Standard + Cross-Groups (`10×10 + Cross`)** | 79,394 | 56.70% | 97.10% |
| **3. Smaller Groups + Cross (`20×5 + Cross`)** | 80,864 | 56.00% | 96.60% |
| **4. Larger Groups + Cross (`5×20 + Cross`)** | 80,294 | **58.00%** | 96.90% |

### 2. Live Public Leaderboard Benchmark (*Enveda CASMI 2026*)
Combining `MultiScaleScannerLayer (5×20 + Cross)` with rational division-multiplication chains (`ParallelDivMulChainLayer`), cross-library training (`ingest_lib`), and neutral-mass dimer decoding (`[2M+H]+`, `[2M+Na]+`):

| Method | Public LB Score (MRR@25) ↑ | Notes |
| :--- | :---: | :--- |
| Standard `0.01 Da` Binned Cosine Floor | `0.091` | Pure CPU library-matching ceiling |
| **HybridMultiScaleDivMulNet (Ours)** | **`0.133`** | **+46% improvement** over standard library baseline |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python benchmark_demo.py
```

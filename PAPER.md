# MultiScaleScannerNet: Hierarchical Global, Group, and Running 1→1 Micro-Scanners for Structured Signal Representations

**Author:** Arsenii Shabrov  
**Date:** November 2026  

## Abstract
Standard dense neural layers apply uniform linear projections across all hidden units, overlooking local cluster structure and multi-scale interactions. We introduce **MultiScaleScannerNet**, a hierarchical neural architecture that partitions a 100-unit hidden layer into $G$ groups of size $S$ ($G \times S = 100$) and processes representations simultaneously across **five observational scales**: (1) a global layer aggregator, (2) a shared global running $1 \to 1$ micro-scanner sweeping across all 100 units, (3) $G$ dedicated group aggregators ($S \to 1$), (4) $G$ group-specific running $1 \to 1$ micro-scanners, and (5) strided cross-group observers linking distant feature clusters. Evaluated on the *Enveda CASMI 2026* mass spectrometry benchmark (2.54M MS2 spectra), a 2-layer architecture with coarser grouping ($5 \times 20$ + Cross-Groups) improves zero-shot Top-1 molecular retrieval on 1,000 unseen molecules from `56.40%` to `58.00%`. When paired with rational division-multiplication chains and neutral-mass dimer decoding, the model achieves **0.133 Public LB** (+46% above the standard `0.091` CPU library-matching baseline).

## 1. Architecture
Each `MultiScaleScannerLayer` transforms an input vector $\mathbf{x} \in \mathbb{R}^{100}$ reshaped into $G$ groups $\mathbf{X}_{\text{grp}} \in \mathbb{R}^{G \times S}$ via five parallel streams:
1. **Global Aggregator ($100 \to 16$):** Captures macro-level layer context.
2. **Global Running Scanner ($1 \to 1$):** Applies a shared micro-MLP element-wise across all 100 units, functioning as a learnable non-linear activation curve.
3. **Group Aggregators ($G \times [S \to 1]$):** Independent sub-networks summarizing each local group into a single scalar verdict.
4. **Group Running Scanners ($G \times [1 \to 1]$):** Group-specific $1 \to 1$ micro-networks providing localized non-linear filtering without quadratic parameter growth.
5. **Cross-Group Observers ($S \times [G \to 1]$):** Strided sub-networks sampling one unit from each group to capture long-range cross-cluster dependencies.

All streams are concatenated into a residual **Layer Arbiter** ($\mathbb{R}^{100}$), stacked across multiple layers, and projected through an **Exit Mini-Network**.

## 2. Experimental Ablation & Competition Benchmark
On 1,000 strictly unseen molecules (`inchikey14`) from *Enveda CASMI 2026*:
* **Base (`10×10`, No Cross-Groups):** `56.40%` Top-1 | `97.20%` Top-25 (`76,414` params)
* **Standard + Cross-Groups (`10×10 + Cross`):** `56.70%` Top-1 | `97.10%` Top-25 (`79,394` params)
* **Smaller Groups + Cross (`20×5 + Cross`):** `56.00%` Top-1 | `96.60%` Top-25 (`80,864` params)
* **Larger Groups + Cross (`5×20 + Cross`):** **`58.00%` Top-1** | `96.90%` Top-25 (`80,294` params)

In live competition evaluation on *Enveda CASMI 2026* (using cross-library `ingest_lib` contrastive training and monomeric/dimeric neutral mass decoding), **HybridMultiScaleDivMulNet** reached **0.133 Public LB**, outperforming the `0.091` binned cosine library-matching baseline by +46%.

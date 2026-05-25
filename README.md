# 🧠 Mini S4 — Structured State Space Sequence Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ygYmLRilaU3NAyAmUHB0QYXRKQnHiEGu)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-S4%20Model-green)

> A complete **pedagogical pipeline** of the S4 (Structured State Space Sequence) model — a powerful alternative to Transformers for long-sequence modeling. Implemented in **NumPy + PyTorch** with full visualizations.

---

## 📌 Project Overview

This project is part of our **Deep Learning course** at the university. We chose **S4** as our topic — a state-space model that combines:

- ✅ **O(L log L)** complexity (like CNNs) instead of O(L²) for Transformers
- ✅ **O(N) memory** (like RNNs) — constant, regardless of sequence length
- ✅ **Full parallelizability** on GPU (like Transformers)
- ✅ **Theoretically infinite context** via HiPPO memory

---

## 🚀 Run the Code

### ▶️ Option 1 — Google Colab (Recommended)

Click the badge above or use this link:

👉 **[Open in Google Colab](https://colab.research.google.com/drive/1ygYmLRilaU3NAyAmUHB0QYXRKQnHiEGu)**

No installation needed — just click **Runtime → Run All**.

### 💻 Option 2 — Run Locally

```bash
# Clone the repository
git clone https://github.com/Ala599/mini-s4-model.git
cd mini-s4-model

# Install dependencies
pip install torch numpy scipy matplotlib

# Run the pipeline
python mini_S4_pipeline.py
```

---

## 📂 Project Structure

```
mini-s4-model/
│
├── mini_S4_pipeline.ipynb     # Main Colab notebook
├── mini_S4_pipeline.py        # Python script version
├── README.md                  # This file
└── s4_pipeline_visualization.png  # Output visualization (generated)
```

---

## 🔬 What This Code Demonstrates

The pipeline covers **9 key steps** of the S4 model:

| # | Section | Description |
|---|---------|-------------|
| 1️⃣ | **ECG Signal Generation** | Synthetic cardiac signal: 3 sinusoids + Gaussian noise |
| 2️⃣ | **HiPPO Matrix** | Mathematically optimal long-range memory via Legendre polynomials |
| 3️⃣ | **ZOH Discretization** | Exact discretization using matrix exponential (scipy) |
| 4️⃣ | **Bilinear Discretization** | Tustin transform — maps jω axis onto unit circle for stability |
| 5️⃣ | **Convolutional Kernel K** | K[l] = C·Ā^l·B̄ — captures dependencies at all distances 0 to L-1 |
| 6️⃣ | **Recurrent Forward Pass** | Step-by-step mode — O(N) memory, ideal for real-time inference |
| 7️⃣ | **Convolutional Forward Pass** | FFT-based parallel mode — O(L log L), ideal for GPU training |
| 8️⃣ | **Trainable S4 Layer (PyTorch)** | Complex diagonal A, multi-channel SSMs, skip connection D |
| 9️⃣ | **ECG Denoising Task** | Training on real signal denoising — 300 epochs, Adam + cosine annealing |

---

## 📊 Output Visualization

The notebook generates a **9-panel figure** saved as `s4_pipeline_visualization.png`:

- 📈 Noisy ECG input signal
- 🌊 Learned convolution kernel K
- 🔁 Recurrent vs Convolutional duality proof (Δ max ≈ 1e-6)
- 🧬 Latent states x(t) evolution (HiPPO memory)
- ✅ ECG denoising result (noisy → clean)
- 📉 Training loss curve (MSE, log scale)
- ⚡ Complexity comparison: S4 O(L log L) vs Transformer O(L²)
- 🎵 Frequency spectrum of kernel K
- 📋 Architecture comparison table

---

## 🧮 Key Mathematical Concepts

### HiPPO Matrix
```
A[n,k] = √(2n+1) · √(2k+1) · (-1)^(n-k)   if k < n
        = n+1                                  if k = n
        = 0                                    otherwise
```

### Discretization (ZOH)
```
Ā = exp(A · Δt)
B̄ = A⁻¹ · (Ā - I) · B
```

### Convolutional Kernel
```
K[l] = C · Ā^l · B̄     for l = 0, 1, ..., L-1
y    = K ★ u             (computed via FFT in O(L log L))
```

---

## ⚙️ Dependencies

```
torch >= 2.0
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

---

## 🤖 AI Tools Used in This Project

| Tool | Usage |
|------|-------|
| **Claude (Anthropic)** | Code generation, explanation, README writing |
| **Gamma.app** | PowerPoint presentation generation |
| **Kling AI / HeyGen** | AI-generated demo videos |
| **Google NotebookLM** | Document synthesis and research |

---



## 📚 References

- Gu, A., Goel, K., & Ré, C. (2022). **Efficiently Modeling Long Sequences with Structured State Spaces**. ICLR 2022. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Gu, A., et al. (2020). **HiPPO: Recurrent Memory with Optimal Polynomial Projections**. NeurIPS 2020. [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)
- The Annotated S4 — [srush.github.io/annotated-s4](https://srush.github.io/annotated-s4/)

---

## 📄 License

This project is for educational purposes as part of a Deep Learning university course.

---


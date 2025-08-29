# Neural Networks: Zero to Hero — Study Notebooks (Learning-in-Public)

A clean, reproducible set of **Jupyter notebooks** following Andrej Karpathy’s *Neural Networks: Zero to Hero* series.  
This repo is designed to be **teachable** (lots of comments, sanity checks, and visuals) and **hackable** (small, modular utilities you can reuse).

> ⚠️ Educational project inspired by the original series. Not affiliated with Andrej Karpathy. Links to the video series and references are provided below.

---

## 🧭 TL;DR

- Run each notebook top-to-bottom and you’ll build intuition **from scalar calculus to tiny language models**.
- **Minimal dependencies**, CPU-friendly, and **seeded for reproducibility**.
- Each chapter includes:
  - Clear objectives & learning outcomes
  - Experiments & toggles you can try (no extra code needed)
  - Notes on failure modes and how to debug them

---

## 📚 Table of Contents

- [What’s in this repo?](#whats-in-this-repo)
- [Repo structure](#repo-structure)
- [Environment & setup](#environment--setup)
- [How to run (VS Code or CLI)](#how-to-run-vs-code-or-cli)
- [Datasets](#datasets)
- [Notebooks & learning outcomes](#notebooks--learning-outcomes)
- [Experiments you can try](#experiments-you-can-try)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [License](#license)

---

## 🔎 What’s in this repo?

A beginner-friendly walkthrough of core deep learning ideas, implemented **from scratch** and then in **PyTorch** when helpful.  
You’ll start with gradients and end up sampling text from a tiny **bigram** model (makemore-style) — seeing exactly how each piece works.

**Why this repo helps you learn fast:**

- *Small, composable steps*: Each cell does one clear thing.
- *Transparent code*: Heavy inline comments and “sanity-check” prints.
- *Visuals where it matters*: Loss curves, heatmaps, top-k tables.
- *Debug-first mindset*: We log shapes, dtypes, and seed runs.

---

## 🗂 Repo structure

```
.
├── 01_Micrograd.ipynb                 # Scalar autograd (Value class), backprop from first principles
├── 02_Makemore_Bigram.ipynb           # Bigram language model: counts, smoothing, sampling, trainable weights
├── data/
│   └── names.txt                      # Simple newline-separated dataset of names (place here)
├── utils/
│   ├── plotting.py                    # (optional) heatmap/loss helpers
│   └── io.py                          # (optional) tiny I/O helpers
├── requirements.txt
└── README.md
```

> Your local filenames may differ (e.g., `02_Chapter2.ipynb`). The README assumes the logical mapping above.

---

## 🧩 Environment & setup

**Python**: 3.9–3.11 recommended (CPU-only is fine).

```bash
# create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
. .venv/Scripts/Activate.ps1
# macOS/Linux (bash/zsh)
source .venv/bin/activate

# upgrade base tools
python -m pip install --upgrade pip

# install deps
pip install -r requirements.txt
```

**Minimal requirements (`requirements.txt`):**
```
numpy
torch
matplotlib
tqdm
jupyter
pandas
```

> Tip: In VS Code, select the `.venv` kernel (bottom-right of the notebook UI) to avoid “module not found” errors.

---

## ▶️ How to run (VS Code or CLI)

**VS Code Notebook UI**
1. Open the repo folder in VS Code.
2. Open a notebook (e.g., `02_Makemore_Bigram.ipynb`).
3. Pick the `.venv` kernel.
4. Run all cells (⏭️).

**CLI (Jupyter)**
```bash
jupyter notebook
# or
jupyter lab
```

---

## 📁 Datasets

Place a simple newline-separated file of names at `data/names.txt`.  
You can use any small text corpus (one item per line). The bigram notebook expects **start/end tokens** to be introduced programmatically; no special markup required.

```text
data/
└── names.txt
```

---

## 📒 Notebooks & learning outcomes

### 01 — Micrograd (from-scratch autograd) ✅
**Core ideas**
- Build a `Value` scalar object with a computation graph.
- Implement forward ops (+, *, tanh, etc.) and **manual backward** (reverse-mode autodiff).
- Visualize simple graphs and confirm **dL/dw** by hand vs. code.

**You’ll be able to:**
- Derive and implement backprop on tiny graphs.
- Inspect gradients at each node.
- Sanity-check gradients with finite differences.

---

### 02 — Makemore (Bigram language model) ✅
**Core ideas**
- Tokenize characters, add **start/end tokens**, build a **V×V bigram count matrix**.
- Apply **Laplace smoothing (α)** to avoid zero-probability rows.
- Convert counts → probabilities; **sample** new names one character at a time.
- Train a **learned bigram** in PyTorch with **cross-entropy** and track loss.

**Why it matters**
- Establish a **baseline** for language modeling that’s easy to debug and extend.
- Build intuition for **n-gram limitations** (no long-range memory).
- Create **interpretable artifacts**: heatmaps, top-k next-char tables.

**Artifacts**
- `bigram_heatmap.png` (optional)
- `samples.txt` (optional)

---

## 🧪 Experiments you can try

- **Smoothing sweep**: vary α ∈ {0, 0.1, 0.5, 1.0} and compare diversity vs. stability.
- **Start/End tokens**: toggle on/off; inspect how endings improve.
- **Train/Val/Test**: confirm generalization; tune training steps/learning rate.
- **Top-k probes**: for each letter, print most likely next characters.
- **Dataset swap**: plug in a different `names.txt` and re-run to see style shifts.

---

## 🔁 Reproducibility

We fix seeds for `random`, `numpy`, and `torch`:

```python
import random, numpy as np, torch
random.seed(1337); np.random.seed(1337); torch.manual_seed(1337)
```

> Note: Exact floating-point results can still vary slightly across OS/PyTorch builds.

---

## 🛠️ Troubleshooting

- **VS Code shows multiple Python interpreters** → pick the one with `.venv` in its path.
- **`ModuleNotFoundError`** → the notebook isn’t using your venv. Re-select kernel and restart.
- **CUDA warnings** → safe to ignore on CPU. The notebooks are optimized for CPU runs.
- **Different samples than the video** → stochastic sampling + dataset differences. Compare your **bigram heatmap** and **train/val loss** to debug.

---

## 🗺️ Roadmap

- [x] 01 — Micrograd: autograd from first principles
- [x] 02 — Makemore — Bigram (counts, smoothing, sampling, trainable bigram)
- [ ] 03 — Makemore — MLP (embedding + simple neural LM)
- [ ] 04 — Recurrent nets (RNN/LSTM intuition)
- [ ] 05 — Attention & Transformers (makemore final)
- [ ] 06 — Extras (tokenization tricks, batching, efficiency)

> The roadmap mirrors the video series at a high level but prioritizes **clarity** and **small steps**.

---

## 🙏 Acknowledgements

- Andrej Karpathy for *Neural Networks: Zero to Hero* — the inspiration for this learning journey.
- **Minfy** and **Blesson Davis** for encouraging a learning-in-public culture and supporting the time to build/share these materials.

---

## 🔗 References

- *Neural Networks: Zero to Hero* — Andrej Karpathy (YouTube series)
- *makemore* repository and related educational materials

---

## 📝 License

This project is for educational purposes. You’re welcome to fork and adapt it; credit is appreciated.



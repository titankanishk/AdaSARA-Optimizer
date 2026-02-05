# AdaSARA Optimizer
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research--Prototype-orange.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

**AdaSARA — Adaptive Speed-Aware Repulsive Adjustment Optimizer**

AdaSARA is a stability-enhanced extension of the Adam optimizer that introduces a dynamic, velocity-aware repulsive correction term. It proactively moderates update magnitudes when both gradient values and parameter update velocity increase simultaneously, improving robustness against instability, overshooting, and oscillations in noisy and ill-conditioned optimization problems.

This repository provides the full implementation, reproducible experiments, convergence plots, and statistical evaluation comparing AdaSARA against Adam, RMSProp, and SGD + Momentum.

---

## 🚀 Key Features
- Speed-aware correction to suppress unstable jumps in gradient updates
- Improved convergence stability in highly curved or noisy landscapes
- Negligible computational overhead vs. Adam (same complexity)
- Demonstrated improvements on Quadratic, Rosenbrock, and MNIST tasks
- Fully reproducible experiment pipeline

---

## 📁 Project Structure

AdaSARA-Optimizer/
│── README.md                     # Project overview & documentation
│── requirements.txt              # Dependencies to reproduce results
│── adasara.py                    # Clean, commented AdaSARA implementation
│── run_experiments.py            # Script to run all evaluations automatically
│
│── experiments/                  # Benchmark experiment scripts
│   ├── quadratic.py
│   ├── rosenbrock.py
│   └── mnist.py

---

## 🧠 Key Idea

AdaSARA introduces a speed-aware correction term:

\[
s_t = \frac{|\Delta_{t-1}| \cdot |g_t|}{|\Delta_{t-1}| + |g_t| + \varepsilon}
\]

which modifies Adam’s denominator:

\[
\Delta_{t}^{AdaSARA} = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + G s_t + \varepsilon}
\]

helping prevent overshooting and oscillations in regions of steep gradient curvature.

---

## 📦 Installation

### **Clone the repository**
```bash
git clone https://github.com/kanishkkhandelwal/AdaSARA-Optimizer.git
cd AdaSARA-Optimizer


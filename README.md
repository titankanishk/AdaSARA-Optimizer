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

```bash
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

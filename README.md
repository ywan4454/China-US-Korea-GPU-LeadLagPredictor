# AI-Quant: China-US-Korea GPU LeadLag Predictor

![Version](https://img.shields.io/badge/Version-v6.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-orange.svg)
![Contributor](https://img.shields.io/badge/Contributor-@ywan4454-red.svg)

> A quantitative trading prediction system for the A-share GPU/AI supply chain, based on "Cross-Market Gravity Transmission" and "Feature Isolation Penalties".

## System Architecture

The global semiconductor and AI supply chains are highly coupled, yet major capital markets operate with significant time differentials. This system leverages these differentials to construct a unidirectional data transmission pipeline:

1. [US Market] (T-1 Close): The global AI bellwether. Closes before the A-share T-day open.
2. [KR Market] (T 10:00 KST): The global memory (HBM) and advanced manufacturing hub. Opens 1 hour before A-shares, reflecting the initial digestion of overnight sentiment.
3. [CN Market] (T 09:30 CST): The global base for AI computing applications, optical modules, and foundry manufacturing. (Prediction Target)

## Core Algorithm Pipeline

### [1] Feature Isolation
- Partitions the A-share AI supply chain into 5 distinct ecological sectors for independent modeling.
- Enforces strict physical feature isolation: Each sector model only has visibility into its corresponding upstream or external market benchmarks, preventing data leakage (e.g., the optical module sector is immune to semiconductor equipment noise).

### [2] Gravity Penalty Mechanism
- Employs a Random Forest 3-class model mapping target returns to Long (>1%), Short (<-1%), and Neutral.
- Introduces an asymmetric trend misclassification penalty (5.0x) via `class_weight={1: 5.0, -1: 5.0, 0: 1.0}`. This heavy penalty forces the model to make decisive non-neutral predictions for high-certainty opportunities, significantly improving sensitivity to market shocks.

### [3] Advanced Feature Engineering
- TAM Weighting: External benchmarks are weighted based on the 2026 Total Addressable Market ceiling of their sub-sectors.
- 3D Momentum: Incorporates 3-day sliding average returns to filter out single-day stochastic noise.

### [4] Regime Shift Filtering
- Training data is strictly truncated post-September 2025 (the period of high global AI and memory resonance) to capture the most authentic current micro-coupling dynamics.

## Sector Coverage

The system clusters 22 leading A-share tickers into 5 ecosystem sectors for macro-level predictions:

- Sector 1: Design Ecology (EDA / GPU / CPU / NPU)
- Sector 2: Manufacturing Ecology (Foundry / Lithography / HBM)
- Sector 3: Packaging & Interconnect (CoWoS / ABF / MSAP)
- Sector 4: Hardware & Deployment (AI Server ODM / Datacenter)
- Sector 5: Connections & Thermal (Optical Modules / Liquid Cooling / Fiber)

## Deployment

Execution is handled via a single CLI entry point (`main.py`), fetching live market data via `yfinance`.

```bash
git clone https://github.com/ywan4454/China-US-Korea-GPU-LeadLagPredictor.git
cd China-US-Korea-GPU-LeadLagPredictor
pip install -r requirements.txt

# Run daily pre-market prediction (Recommended at 09:00 CST)
source venv/bin/activate
python main.py
```

## Contributors

* Core Architect / Quant Developer: [@ywan4454](https://github.com/ywan4454)

### Contribution Workflow
1. Fork the repository.
2. `git checkout -b feature/your-feature-name`
3. `git commit -m 'feat: add new momentum factor'`
4. `git push origin feature/your-feature-name`
5. Open a Pull Request.

---
Disclaimer: This system is based on quantitative statistical modeling. Historical patterns do not guarantee future performance. The model relies solely on price and volume signals and does not account for fundamental shifts or black swan events. Not financial advice.

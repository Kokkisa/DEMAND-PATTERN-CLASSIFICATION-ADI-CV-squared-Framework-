# DEMAND-PATTERN-CLASSIFICATION-ADI-CV-squared-Framework-
Classify SKUs into demand categories (smooth, erratic, lumpy, intermittent) to select the right forecasting meth# Demand Pattern Classification (ADI / CV2 Framework)

## Overview
Demand planning project that classifies 100 SKUs into four demand categories (Smooth, Erratic, Intermittent, Lumpy) using the ADI/CV2 framework, then recommends the optimal forecasting method for each pattern. Demonstrates differentiated forecasting strategy based on demand characteristics.

**Built by:** Nithin Kumar Kokkisa — Senior Demand Planner with 12+ years at HPCL managing 180,000 MTPA facility.

---

## Business Problem
Using one forecasting method for all products is a common mistake. ARIMA works well for smooth demand but fails on intermittent demand (many zero periods). Croston's method handles intermittent patterns but is unnecessary for regular demand. This project classifies each SKU by its demand pattern and assigns the right method — improving overall forecast accuracy while optimizing planner effort.

## The ADI/CV2 Framework

|  | CV2 < 0.49 (Low variability) | CV2 >= 0.49 (High variability) |
|---|---|---|
| **ADI < 1.32** (Regular) | **Smooth** — Easy to forecast | **Erratic** — Variable quantity |
| **ADI >= 1.32** (Sporadic) | **Intermittent** — Sporadic timing | **Lumpy** — Hardest to forecast |

**ADI** = Number of periods / Periods with demand (higher = more sporadic)

**CV2** = (std / mean)^2 of non-zero demands (higher = more variable)

## Forecasting Method by Pattern

| Pattern | Recommended Method | Target MAPE |
|---------|-------------------|-------------|
| Smooth | ARIMA / ETS / Prophet | 5-15% |
| Erratic | Damped ETS / Robust ARIMA | 15-25% |
| Intermittent | Croston's / SBA | 25-40% |
| Lumpy | Croston's / TSB / Aggregate | 30-50%+ |

## Visualizations

| Chart | Insight |
|-------|---------|
| ADI vs CV2 Scatter | THE classification chart -- 4 quadrants with thresholds |
| Pattern Examples | Sample demand time series for each category |
| Metric Distributions | ADI and CV2 histograms with threshold lines |
| Portfolio Composition | SKU count vs demand volume by pattern |

## Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)

---

## About
Part of a **30-project data analytics portfolio**. Completes Block A (Core Demand Planning): Prophet, ARIMA comparison, Forecast Accuracy Dashboard, Hierarchical Forecasting, and Demand Pattern Classification.
[README_project13.md](https://github.com/user-attachments/files/25788782/README_project13.md)
od for each.

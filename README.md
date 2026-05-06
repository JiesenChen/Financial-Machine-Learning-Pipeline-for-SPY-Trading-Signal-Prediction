# Volatility-Adjusted Signal Learning and Rolling IC Feature Selection
## A Refined Financial Machine Learning Framework for Next-Day SPY Trading Decisions
#### STAT GR5243 — Applied Data Science Project, Columbia University  
*May 2026*
---
## Overview

This project develops a machine learning pipeline for predicting **next-day Long / Flat / Short trading signals for SPY**, the ETF tracking the S&P 500 Index. Instead of forecasting the exact next-day price level, the project frames return prediction as a practical three-class trading decision problem: go long, stay flat, or go short.
The project evaluates whether publicly available market, macroeconomic, technical, intraday, volume, calendar, regime, and cross-market information can generate more stable and risk-aware SPY trading signals. Rather than focusing only on raw classification accuracy, the workflow emphasizes **signal quality**, **leakage-aware validation**, and **financially interpretable feature selection**.

**Core research questions:**

1. Can refined financial features improve the stability of next-day SPY directional signals?
2. Can volatility-normalized return labels produce more meaningful Long / Flat / Short classifications across changing market volatility regimes?
3. Can Rolling Spearman IC ranking, category-aware screening, and correlation-based clustering reduce noisy or redundant predictors while preserving useful trading information?
4. Do the resulting selected features provide a stronger foundation for later trading-based evaluation, including cumulative return, Sharpe ratio, maximum drawdown, and false Long risk?

**Key contribution:** This project moves beyond a simple classification model by building a refined financial machine learning workflow that combines expanded feature engineering, volatility-normalized label construction, walk-forward validation design, Rolling IC feature ranking, category-aware feature screening, hierarchical correlation cleanup, and final selected-feature construction for next-day SPY trading signal prediction.
---
## Methodology

The framework is organized as a multi-stage financial machine learning pipeline:

### Stage 1 — Expanded Financial Feature Engineering

Construct a broad candidate feature pool from SPY returns, technical indicators, volatility measures, macro-risk variables, yield curve changes, intraday structure, volume pressure, calendar effects, VIX regimes, and cross-market ETF returns.

### Stage 2 — Volatility-Adjusted Label Construction

Transform next-day SPY returns into Long / Flat / Short trading labels using volatility-normalized return information. This makes the classification target more adaptive to changing market volatility conditions than a fixed return threshold.

### Stage 3 — Rolling IC Feature Selection and Correlation Cleanup

Compute Rolling Spearman Information Coefficients on the train/CV period only to identify features with more stable predictive relationships. Then apply category-aware screening and correlation-based cleanup to reduce redundant predictors.

### Stage 4 — Signal Modeling and Trading-Based Evaluation

Train classification models on the selected features and evaluate the resulting signals using both machine learning metrics and trading metrics, including cumulative return, Sharpe ratio, maximum drawdown, and false Long risk.

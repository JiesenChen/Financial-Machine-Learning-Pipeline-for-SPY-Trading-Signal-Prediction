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

**Key contribution:** This project builds a leakage-aware financial machine learning workflow that combines expanded feature engineering, volatility-adjusted label construction, walk-forward validation design, Rolling IC feature ranking, category-aware screening, and correlation-based feature cleanup for next-day SPY trading signal prediction.

---

## Business Problem

The business problem is to convert noisy short-horizon SPY return information into a practical next-day trading decision. Instead of predicting the exact next-day SPY price, this project studies whether public market data can support a disciplined **Long / Flat / Short** signal.

The current modeling dataset contains **1,027 daily observations** after feature construction and missing-value removal. The data are split chronologically into **821 train/CV observations** and **206 final test observations**, so future observations are not used during label construction, feature selection, or validation.

A key challenge is that daily SPY returns are noisy and tail-sensitive. In the train/CV period, next-day log returns have an average close to **0.04%**, a standard deviation around **1.17%**, and a wide observed range from approximately **-6.03%** to **+9.99%**. Because many daily return movements are small relative to volatility, the project does not force the model to trade every day.

The core business question is therefore:

> Can a volatility-adjusted, leakage-aware, and feature-selected machine learning pipeline produce more stable and interpretable next-day SPY trading signals than a raw feature classification approach?

---

## Methodology

The framework is organized as a multi-stage financial machine learning pipeline:

### Stage 1 — Expanded Financial Feature Engineering

Construct a broad candidate feature pool from SPY returns, technical indicators, volatility measures, macro-risk variables, yield curve changes, intraday structure, volume pressure, calendar effects, VIX regimes, and cross-market ETF returns.

### Stage 2 — Volatility-Adjusted Label Construction

Normalize next-day SPY returns by 20-day realized volatility and define Long / Flat / Short labels using train/CV-only quantile cutoffs.

### Stage 3 — Walk-Forward Validation Design

Use a chronological train/test split and expanding-window validation inside the train/CV period to preserve the time-series structure and reduce look-ahead bias.

### Stage 4 — Rolling IC Feature Selection

Compute Rolling Spearman Information Coefficients on train/CV only, then apply category-aware screening and correlation-based cleanup to select stable and non-redundant predictors.

### Stage 5 — Modeling and Trading Evaluation

Use the final selected features to train classification models and evaluate signals using both machine learning metrics and trading-based metrics.

---

## Data Sources

The project uses daily market and macro-related data aligned to SPY trading dates.

| Data Group | Variables / Instruments | Purpose |
|---|---|---|
| SPY OHLCV | Open, High, Low, Close/Price, Volume | Main asset and technical feature construction |
| Market risk | VIX | Equity market volatility and risk regime |
| Macro-sensitive assets | DXY, TNX, OIL | Dollar, rates, and oil market conditions |
| Treasury curve | DGS2, DGS10 | 2Y–10Y yield curve features |
| Cross-market ETFs | QQQ, IWM, DIA | U.S. style and sector-risk proxies |
| Calendar | NYSE trading calendar | Weekday and month-boundary effects |

---

## Feature Engineering

The full candidate feature pool contains **81 candidate features**. These features are grouped into several financial categories:

| Feature Group | Examples |
|---|---|
| Lagged returns | `log_ret`, `ret_lag_2`, `ret_lag_3`, `ret_lag_5`, `ret_lag_10` |
| Trend indicators | `SMA_gap_5`, `SMA_gap_10`, `SMA_gap_20`, `SMA_gap_50` |
| Momentum indicators | `RSI_14`, `MACD_hist` |
| Volatility indicators | `Volatility_5`, `Volatility_10`, `Volatility_20`, `Volatility_60` |
| Bollinger features | `BB_width`, `BB_percent_b` |
| Macro changes | VIX, DXY, TNX, OIL return/change features |
| Macro z-scores | `VIX_z_20`, `VIX_z_60`, `DXY_z_20`, `DXY_z_60`, etc. |
| Yield curve | `YC_2Y10Y_change_1`, `YC_2Y10Y_z_20`, `YC_2Y10Y_z_60` |
| Intraday structure | `intraday_ret`, `overnight_gap`, `high_low_range`, `close_location_value` |
| Volume pressure | `relative_volume_20`, `volume_z_20`, `volume_ret_interaction` |
| Calendar effects | `is_monday`, `is_friday`, `is_month_start`, `is_month_end` |
| VIX regime | `vix_regime_high`, `vix_regime_low` |
| Cross-market ETFs | QQQ, IWM, DIA lagged return features |

After feature construction and missing-value removal, the final `model_data` contains **1,027 observations** and **0 missing values**.

---

## Target and Label Design

The raw response variable is next-day SPY log return:

```text
next_ret = log(Price_{t+1} / Price_t)
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
---
## Business Problem

The business problem is to convert noisy short-horizon SPY return information into a practical next-day trading decision. Instead of forecasting the exact next-day SPY price, this project studies whether publicly available daily market data can support a disciplined **Long / Flat / Short** signal for SPY.

In the current notebook, the final modeling dataset contains **1,027 daily observations** after feature construction and missing-value removal. The data are split chronologically into:

| Split | Observations | Period |
|---|---:|---|
| Train/CV | 821 | 2022-03-30 to 2025-07-09 |
| Final Test | 206 | 2025-07-10 to 2026-05-04 |

This time-based split is central to the business problem. A trading signal must be evaluated as a forward-looking decision, so future observations cannot be used when constructing labels, selecting features, or validating the model.

A key challenge is that next-day SPY returns are highly noisy. In the train/CV period, the next-day log return has an average close to **0.04%**, with a standard deviation around **1.17%**. The observed next-day return range is wide, from approximately **-6.03%** to **+9.99%**. This means that most daily return movements are small relative to the noise level, while occasional tail events can dominate trading performance.

Because of this, the project does not force the model to trade every day. Instead, it transforms next-day return prediction into a three-class decision problem:

| Signal | Trading Interpretation |
|---|---|
| Long | Take positive SPY exposure |
| Flat | Stay out of the market |
| Short | Take negative SPY exposure |

The current code uses a **volatility-adjusted label design**. Specifically, the next-day return is normalized by 20-day realized volatility:

```text
next_ret_vol_adj = next_ret / Volatility_20
---
## Methodology

The framework is organized as a multi-stage financial machine learning pipeline. The current version focuses on constructing a refined modeling dataset, building volatility-adjusted trading labels, selecting stable and non-redundant predictors, and preparing the final feature matrix for later model training and trading evaluation.

### Stage 1 — Expanded Financial Feature Engineering

The first stage constructs a broad candidate feature pool from daily SPY market data and related public market variables. The raw data include SPY OHLCV variables, market risk proxies, macro-sensitive assets, Treasury yield data, and U.S. style ETF proxies.

The initial feature engineering process creates **50 base candidate features**, including:

| Feature Group | Features / Examples | Purpose |
|---|---|---|
| Lagged returns | `log_ret`, `ret_lag_2`, `ret_lag_3`, `ret_lag_5`, `ret_lag_10` | Capture short-term return momentum or reversal |
| Trend indicators | `SMA_gap_5`, `SMA_gap_10`, `SMA_gap_20`, `SMA_gap_50` | Measure price deviation from moving-average trend |
| Momentum indicators | `RSI_14`, `MACD_hist` | Capture overbought/oversold and momentum strength |
| Volatility indicators | `Volatility_5`, `Volatility_10`, `Volatility_20`, `Volatility_60` | Measure realized volatility across multiple horizons |
| Bollinger features | `BB_width`, `BB_percent_b` | Capture volatility band width and price position |
| Macro changes | VIX, DXY, TNX, OIL return/change features with lags 1, 2, 3, 5, 10, 20 | Capture market risk, dollar, rates, and oil shocks |
| Macro z-scores | `VIX_z_20`, `VIX_z_60`, `DXY_z_20`, `DXY_z_60`, `TNX_z_20`, `TNX_z_60`, `OIL_z_20`, `OIL_z_60` | Normalize macro variables relative to recent rolling states |

The feature set is then expanded with additional market-structure and regime variables. After adding yield curve, intraday, volume, calendar, and VIX regime features, the candidate feature count increases from **50 to 66**.

| Additional Feature Group | Features / Examples | Purpose |
|---|---|---|
| Yield curve | `YC_2Y10Y_change_1`, `YC_2Y10Y_z_20`, `YC_2Y10Y_z_60` | Capture 2Y–10Y Treasury curve changes and abnormal curve states |
| Intraday structure | `intraday_ret`, `overnight_gap`, `high_low_range`, `close_location_value` | Capture within-day strength, overnight movement, daily range, and close location |
| Volume pressure | `relative_volume_20`, `volume_z_20`, `volume_ret_interaction` | Measure abnormal trading volume and return-volume interaction |
| Calendar effects | `is_monday`, `is_friday`, `is_month_start`, `is_month_end` | Capture weekday and month-boundary effects based on NYSE trading calendar |
| VIX regime | `vix_regime_high`, `vix_regime_low` | Identify high- and low-volatility market regimes using rolling VIX quantiles |

The code also adds U.S. cross-market style ETF features using QQQ, IWM, and DIA. For each ETF, 1-, 2-, 3-, 5-, and 10-day log return features are created. This adds **15 cross-market features**:

```text
QQQ_ret_1, QQQ_ret_2, QQQ_ret_3, QQQ_ret_5, QQQ_ret_10
IWM_ret_1, IWM_ret_2, IWM_ret_3, IWM_ret_5, IWM_ret_10
DIA_ret_1, DIA_ret_2, DIA_ret_3, DIA_ret_5, DIA_ret_10
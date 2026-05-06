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
```

Then, the Long / Flat / Short labels are created using train/CV quantile cutoffs of this volatility-adjusted target. This design makes the label more adaptive to different volatility regimes than a raw return threshold.

The resulting label distribution is intentionally conservative:

| Split | Short | Flat | Long |
|---|---:|---:|---:|
| Train/CV | 206 / 25.1% | 409 / 49.8% | 206 / 25.1% |
| Final Test | 51 / 24.8% | 100 / 48.5% | 55 / 26.7% |

This reflects the trading objective: the model should only generate Long or Short signals when the return is meaningfully positive or negative relative to the current volatility environment. Otherwise, the appropriate decision is Flat.

Another major business challenge is feature redundancy. The project begins with **81 candidate features** across multiple financial information groups:

| Feature Group | Examples |
|---|---|
| Lagged returns | `log_ret`, multi-day return lags |
| Trend indicators | `SMA_gap_5`, `SMA_gap_10`, `SMA_gap_20`, `SMA_gap_50` |
| Momentum indicators | `RSI_14`, `MACD_hist` |
| Volatility indicators | `Volatility_5`, `Volatility_10`, `Volatility_20`, `Volatility_60` |
| Bollinger / price position | `BB_width`, `BB_percent_b` |
| Macro changes | VIX, DXY, TNX, OIL return/change features |
| Macro normalization | rolling z-score features such as `VIX_z_20`, `DXY_z_60` |
| Yield curve | `YC_2Y10Y_change_1`, `YC_2Y10Y_z_20`, `YC_2Y10Y_z_60` |
| Intraday structure | `intraday_ret`, `overnight_gap`, `high_low_range`, `close_location_value` |
| Volume pressure | `relative_volume_20`, `volume_z_20`, `volume_ret_interaction` |
| Calendar effects | weekday and month-start/month-end indicators |
| VIX regime | high- and low-volatility regime indicators |
| Cross-market ETFs | QQQ, IWM, DIA, EEM, VGK, EWJ return features |

Many of these variables are economically meaningful but highly correlated. For example, lagged returns, SMA gaps, volatility windows, macro z-scores, and cross-market ETF returns can contain overlapping information. Using all of them directly can increase model noise and reduce interpretability.

To address this, the current code applies a train/CV-only feature selection process. It first ranks candidate features using **Rolling Spearman Information Coefficient (IC)** against the volatility-adjusted next-day return target. It then applies category-aware screening and correlation-based cleanup to reduce redundant predictors. After this process, the modeling feature set is reduced from **81 candidate features** to **21 selected features**.

Therefore, the core business problem is not simply to maximize classification accuracy. The project is designed to answer a more practical financial question:

> Can a volatility-adjusted, leakage-aware, and feature-selected machine learning pipeline produce more stable and interpretable next-day SPY trading signals than a raw feature classification approach?

This framing aligns the machine learning task with the actual trading decision: avoid overreacting to noisy daily price movements, preserve economically meaningful signals, reduce redundant predictors, and prepare the final selected signal set for out-of-sample trading evaluation.

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
```

After all feature construction steps, the full candidate feature pool contains **81 candidate features**. The final `model_data` contains **1,027 observations** and **82 columns**, including 81 candidate features plus the next-day return target. After missing-value removal, there are **0 missing values**.

---

### Stage 2 — Chronological Train/Test Split

The second stage creates a chronological final holdout split. Because the project is designed for forward-looking trading decisions, the data are not randomly shuffled.

The final modeling dataset is split into:

| Split | Observations | Share | Date Range |
|---|---:|---:|---|
| Train/CV | 821 | 79.94% | 2022-03-30 to 2025-07-09 |
| Final Test | 206 | 20.06% | 2025-07-10 to 2026-05-04 |

The train/CV period is used for label threshold estimation, feature selection, and walk-forward validation design. The final test period is held out for later out-of-sample model evaluation.

---

### Stage 3 — Volatility-Adjusted Label Construction

The third stage converts next-day SPY return prediction into a three-class trading signal problem. The raw response variable is the next-day SPY log return:

```text
next_ret = log(Price_{t+1} / Price_t)
```

Instead of labeling observations directly from raw next-day returns, the current code constructs a volatility-adjusted target:

```text
next_ret_vol_adj = next_ret / Volatility_20
```

Here, `Volatility_20` is the 20-day realized volatility computed from past and current SPY log returns. This adjustment makes the target more comparable across different volatility regimes.

The Long / Flat / Short labels are then defined using train/CV-only quantile cutoffs of `next_ret_vol_adj`:

| Threshold | Value | Interpretation |
|---|---:|---|
| `q_low` | -0.515819 | Short threshold |
| `q_high` | 0.723424 | Long threshold |

The label rule is:

```text
label = -1  if next_ret_vol_adj <= q_low
label =  1  if next_ret_vol_adj >= q_high
label =  0  otherwise
```

The resulting class distributions are:

| Split | Short (-1) | Flat (0) | Long (1) |
|---|---:|---:|---:|
| Train/CV | 206 / 25.09% | 409 / 49.82% | 206 / 25.09% |
| Final Test | 51 / 24.76% | 100 / 48.54% | 55 / 26.70% |

This label design is intentionally conservative. Roughly half of the observations are assigned to the Flat class, meaning the framework is not designed to trade every day. Long and Short signals are reserved for observations where next-day returns are large relative to the recent volatility environment.

---

### Stage 4 — Walk-Forward Validation Design

The fourth stage defines an expanding-window walk-forward validation structure within the train/CV period. The final test set is not used in this step.

The current code uses:

```text
TimeSeriesSplit(n_splits = 5, gap = 1)
```

This produces five expanding train/validation folds:

| Fold | Train Period | Train Size | Validation Period | Validation Size |
|---:|---|---:|---|---:|
| 1 | 2022-03-30 to 2022-10-18 | 140 | 2022-10-20 to 2023-05-05 | 136 |
| 2 | 2022-03-30 to 2023-05-04 | 276 | 2023-05-08 to 2023-11-17 | 136 |
| 3 | 2022-03-30 to 2023-11-16 | 412 | 2023-11-20 to 2024-06-05 | 136 |
| 4 | 2022-03-30 to 2024-06-04 | 548 | 2024-06-06 to 2024-12-18 | 136 |
| 5 | 2022-03-30 to 2024-12-17 | 684 | 2024-12-19 to 2025-07-09 | 136 |

This structure preserves time order and introduces a one-day gap between training and validation to reduce look-ahead risk.

---

### Stage 5 — Rolling IC Feature Ranking

The fifth stage ranks all candidate features using Rolling Spearman Information Coefficient (IC) on the train/CV period only. The target used for IC ranking is the volatility-adjusted target:

```text
target_col = next_ret_vol_adj
```

The Rolling IC settings are:

| Parameter | Value |
|---|---:|
| Rolling window | 60 trading days |
| Minimum periods | 40 observations |
| Candidate features | 81 |
| Rolling IC summary shape | 81 × 8 |

For each feature, the code computes rolling Spearman rank correlation against `next_ret_vol_adj`, then summarizes:

```text
mean_IC
abs_mean_IC
IC_std
IC_IR
abs_IC_IR
positive_IC_share
non_missing_windows
```

The strongest Rolling IC features by absolute IC information ratio include:

| Rank | Feature | mean_IC | abs_mean_IC | IC_IR | positive_IC_share |
|---:|---|---:|---:|---:|---:|
| 1 | `SMA_gap_50` | -0.142587 | 0.142587 | -1.270428 | 0.101050 |
| 2 | `SMA_gap_20` | -0.102809 | 0.102809 | -1.029259 | 0.135171 |
| 3 | `high_low_range` | 0.087656 | 0.087656 | 1.001084 | 0.853018 |
| 4 | `RSI_14` | -0.117210 | 0.117210 | -0.983871 | 0.200787 |
| 5 | `VIX` | 0.092523 | 0.092523 | 0.840320 | 0.780840 |

This stage does not finalize the modeling features. It provides a train/CV-only ranking of candidate predictors based on the stability and direction of their relationship with the volatility-adjusted next-day target.

---

### Stage 6 — Category-Aware Feature Screening

The sixth stage applies category-aware screening instead of selecting features globally from a single ranked list. This is important because many features within the same economic category are highly correlated. For example, different SMA gaps, volatility windows, return lags, and ETF return horizons may capture overlapping information.

The code first screens features within comparable groups, including:

```text
Lagged Return
Trend
Volatility
VIX Return Lags
DXY Return Lags
TNX Change Lags
OIL Return Lags
VIX Z-score
DXY Z-score
TNX Z-score
OIL Z-score
Yield Curve
QQQ Return Lags
IWM Return Lags
DIA Return Lags
Intraday
Volume
Calendar
VIX Regime
```

Within each group, the top features are selected based on Rolling IC strength. Examples of group-level selected signals include:

| Group | Representative Selected Feature(s) |
|---|---|
| Lagged Return | `ret_lag_5`, `ret_lag_10` |
| Trend | `SMA_gap_50`, `SMA_gap_20` |
| Volatility | `Volatility_60`, `Volatility_20` |
| VIX Return Lags | `VIX_ret_20`, `VIX_ret_5` |
| DXY Return Lags | `DXY_ret_1`, `DXY_ret_2` |
| TNX Change Lags | `TNX_change_5`, `TNX_change_10` |
| OIL Return Lags | `OIL_ret_1`, `OIL_ret_20` |
| VIX Z-score | `VIX_z_60`, `VIX_z_20` |
| Yield Curve | `YC_2Y10Y_z_60`, `YC_2Y10Y_z_20` |
| Intraday | `high_low_range`, `overnight_gap` |
| Volume | `volume_ret_interaction` |
| Calendar | `is_friday` |
| VIX Regime | `vix_regime_low` |

This step keeps the feature selection process economically diversified instead of allowing one highly correlated group to dominate the final model.

---

### Stage 7 — Correlation-Based Clustering and Redundancy Cleanup

The seventh stage reduces redundancy among screened features using correlation-based clustering and manual cleanup.

After cluster-based screening, the selected feature list contains **23 features**. The code then checks absolute pairwise correlations among these selected features. Before final manual cleanup, the remaining high-correlation pairs above the threshold are:

| Feature 1 | Feature 2 | Absolute Correlation |
|---|---|---:|
| `ret_lag_5` | `DIA_ret_5` | 0.916308 |
| `SMA_gap_50` | `RSI_14` | 0.887349 |
| `SMA_gap_20` | `RSI_14` | 0.869225 |

The threshold used in the code is:

```text
abs_corr >= 0.85
```

To reduce redundancy, the code manually removes:

```text
DIA_ret_5
SMA_gap_20
```

After this final cleanup, the feature set is reduced from **23 to 21 selected features**.

The final selected feature list is:

```text
SMA_gap_50
high_low_range
RSI_14
VIX
MACD_hist
ret_lag_5
Volatility_60
Volatility_20
vix_regime_low
VIX_z_60
OIL_ret_1
overnight_gap
TNX_change_5
volume_ret_interaction
DXY_ret_1
TNX_z_60
VIX_ret_20
OIL_z_60
is_friday
YC_2Y10Y_z_60
DXY_z_20
```

A final correlation check shows that only one pair remains above the 0.85 threshold:

| Feature 1 | Feature 2 | Absolute Correlation |
|---|---|---:|
| `SMA_gap_50` | `RSI_14` | 0.887349 |

This pair is retained because the two variables represent different financial interpretations: one captures price-trend deviation relative to a 50-day moving average, while the other captures momentum/overbought-oversold conditions.

---

### Stage 8 — Final Modeling Dataset Construction

The final stage constructs the train/test matrices using only the selected features.

The final modeling matrices are:

| Matrix | Shape |
|---|---:|
| `X_train_cv` | 821 × 21 |
| `y_train_cv` | 821 |
| `X_test` | 206 × 21 |
| `y_test` | 206 |

The final data quality checks confirm:

| Check | Train/CV | Test |
|---|---:|---:|
| Missing values | 0 | 0 |
| Infinite values | 0 | 0 |
| Selected feature count | 21 | 21 |

At this point, the notebook has completed the data preparation, volatility-adjusted label construction, walk-forward validation design, Rolling IC feature ranking, category-aware screening, correlation cleanup, and final X/y construction. The next step is to train classification models on the selected features and evaluate the resulting signals using both machine learning metrics and trading-based metrics.
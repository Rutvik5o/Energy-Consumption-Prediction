# Energy Consumption Forecasting — TSO Dataset

by **Rutvik Prajapati**

---

so this is my submission for the energy forecasting task. took me a while to figure out the right approach but i think what i ended up with is actually pretty interesting compared to just doing the usual thing everyone does.

the dataset is ~52,000 hourly electricity consumption readings from a Transmission System Operator (TSO), covering 2016 to 2021. the goal was to build a model that can forecast future hourly energy demand.

---

## what i did differently

honestly my first instinct was to just train an LSTM on the raw values. but then i sat with the data for a bit and realized — energy consumption has this very obvious layered structure. there's a daily cycle, a weekly cycle, and a yearly seasonal cycle all stacked on top of each other. if you just feed raw numbers to a neural net, it has to learn all of that from scratch which is kind of wasteful.

so instead i went with:

**STL Decomposition → LightGBM on residuals → Quantile regression for uncertainty**

the idea is: use STL (Seasonal-Trend decomposition using LOESS) to peel off the cycles we already know exist, then let the ML model focus only on what's left (the residual). that residual is WAY easier to model because it has no trend and no seasonal swing.

and then i used quantile regression inside LightGBM to get prediction intervals (10th and 90th percentile) — because in energy management you dont just want "what will demand be" you also want "how bad could it get". that's what reserve capacity planning is based on.

---

## project structure

```
.
├── energy_forecasting.ipynb    # main notebook — run this
├── data.xlsm                   # raw TSO data (put in same folder)
├── model_main_lgbm.pkl         # saved point forecast model
├── model_q10_lgbm.pkl          # saved 10th percentile model
├── model_q90_lgbm.pkl          # saved 90th percentile model
├── feature_cols.json           # feature list (needed for inference)
├── test_predictions.csv        # model output on 2021 test set
├── fig1_overview.png
├── fig2_seasonal_cycles.png
├── fig3_fft.png
├── fig4_heatmap_yoy.png
├── fig5_stl_decomp.png
├── fig6_error_analysis.png
├── fig7_forecast_results.png
├── fig8_feature_importance.png
└── fig9_insights.png
```

---

## how to run

1. clone the repo / download the files
2. put `data.xlsm` in the same folder as the notebook
3. install dependencies (see below)
4. open `energy_forecasting.ipynb` and run all cells top to bottom

thats it. every cell has comments explaining what its doing and why.

### dependencies

```bash
pip install pandas numpy matplotlib seaborn lightgbm statsmodels scikit-learn openpyxl scipy joblib
```

i used python 3.10 but anything 3.8+ should work fine.

---

## approach walkthrough

### step 1 — data loading & cleaning

the data comes as a `.xlsm` (excel with macros). loaded it with openpyxl. 3 columns: start time UTC, end time UTC, consumption in MWh.

first thing i checked was whether the timestamps are perfectly 1-hour apart. they're not — there are gaps from DST transitions and what looks like data outages. the nastiest one is a 15-hour gap in April 2016.

**gap filling strategy:**
- for gaps ≤ 3 hours: linear time interpolation (works fine for small gaps)
- for larger gaps: find same hour + same day-of-week from surrounding 4 weeks and use the median. this is much better than linear for seasonal data because a 15h gap on a Tuesday morning should look like other Tuesday mornings, not like a straight line between the values before and after

then ran a rolling modified z-score outlier check (Iglewicz & Hoaglin method, threshold=7). conservative threshold because energy has real extreme events (heatwaves etc) that look like outliers but aren't.

### step 2 — EDA

did 4 types of analysis:

- **full series plot**: shows the 6-year trend + COVID dip in 2020 clearly
- **seasonal cycle plots**: daily, weekly, yearly — all three are clearly structured. weekdays vs weekends are noticeably different in the hourly profile
- **FFT spectral analysis**: mathematically confirms the dominant periods at 24h, 168h (7 days), and ~8760h (1 year). this isn't just "looks seasonal" — it's a proper frequency analysis
- **hour × day-of-week heatmap**: shows which combinations are peak vs off-peak

### step 3 — STL decomposition

ran STL with `period=168` (weekly), `robust=True` (resistant to outliers).

after decomposition:
- seasonal component explains ~60% of total variance
- trend explains another ~25%
- residual (what ML models): only ~15% of variance

so the model only needs to explain 15% of the original signal's variance. that's the point.

### step 4 — feature engineering

built ~45 features:

| feature type | examples | why |
|---|---|---|
| lag features | lag_1h, lag_24h, lag_168h, lag_336h | recent history context |
| rolling stats | roll_mean_24h, roll_std_168h | momentum + volatility |
| fourier encoding | hour_sin, hour_cos, dow_sin, doy_cos | cyclic distance (hour 23 and 0 are close) |
| STL components | trend_feature, seasonal_feature | where are we in the cycle |
| delta features | delta_1h, delta_24h, delta_168h | rate of change signals |
| calendar | is_weekend, is_morning, is_evening | domain knowledge |
| interactions | hour × is_weekend | weekend hourly pattern ≠ weekday |

the single most important feature ended up being `lag_168h` (same hour last week) which makes total sense — energy is extremely periodic.

### step 5 — model training

trained 3 LightGBM models:
- **main model** (`objective='regression'`, MAE loss): point forecast
- **q10 model** (`objective='quantile', alpha=0.10`): lower bound
- **q90 model** (`objective='quantile', alpha=0.90`): upper bound

used TimeSeriesSplit with 5 folds and a 24h gap between train and validation to prevent leakage. early stopping on each fold.

train set: 2016–2020 (~44,000 hours)
test set: 2021 (~8,700 hours)

### step 6 — evaluation

| metric | our model | naive baseline (last-week same hour) |
|---|---|---|
| MAE (MWh) | ~180 | ~320 |
| MAPE (%) | ~1.9% | ~3.4% |
| R² | ~0.975 | ~0.93 |
| 80% PI coverage | ~80% | — |

improvement over naive: roughly 44% better MAE. the naive last-week baseline is actually a pretty strong competitor for energy data so beating it by this margin is meaningful.

also ran an Augmented Dickey-Fuller test on the forecast residuals — they came out stationary (p < 0.05) which means the model has captured all the systematic structure. what's left is just noise.

---

## key findings

**1. lag-168h dominates**
the most important feature is consumption from exactly 168 hours ago (same hour last week). energy demand follows weekly routines incredibly consistently.

**2. transition hours are hardest**
model error is highest at 7-9am (morning ramp-up) and 17-19pm (evening peak). these are the hours when demand can change fastest and depends most on human behavior which is harder to predict.

**3. 2020 COVID demand drop**
visible in both the raw data and the year-over-year boxplot. mean consumption dropped noticeably in Mar-May 2020. the model adapts to this through its rolling features — it doesn't need explicit knowledge of COVID, the recent lag values implicitly capture the regime shift.

**4. prediction intervals are well calibrated**
80% PI (q10 to q90) achieved ~80% actual coverage on the test set. upper and lower violations are roughly symmetric (~10% each). this means the uncertainty estimates are trustworthy, not just wide.

**5. spectral analysis confirms the approach**
FFT on the raw series shows three sharp peaks at exactly 1 day, 7 days, and ~1 year. these are deterministic components that should be modeled explicitly (via STL), not left for the ML model to figure out.

---

## limitations & what i'd do with more time

- **no weather data**: temperature is the biggest driver of energy demand (heating/cooling) but i didn't have it. adding hourly temperature would probably cut error by another 30-40%
- **no holiday calendar**: public holidays behave like Sundays but the model doesn't know which days are holidays. easy fix — add a boolean `is_public_holiday` feature
- **the 15h gap in April 2016**: my seasonal fill handles it but it's a big gap. if this were production, i'd flag these periods and exclude them from training evaluation
- **COVID regime shift**: the model eventually adapts through rolling lags but the first few weeks of lockdown would have large errors. a simple regime detection flag would help
- **hyperparameter tuning**: i set params manually based on intuition. proper Optuna or Bayesian search would squeeze out more performance

---

## why not LSTM?

i get asked this so i'll explain it here:

LSTMs need to learn seasonality implicitly from data — this takes a lot of data and training time. for a structured tabular time series with *known* seasonal periods, gradient boosted trees with explicit lag/fourier features consistently outperform LSTMs and are much faster to train and easier to interpret. this was also empirically shown in the M5 forecasting competition where LGBM-based solutions dominated the leaderboard.

if i had multi-variate data (energy + weather + prices + load type etc) then a Temporal Fusion Transformer would be worth looking at. but for pure univariate with explicit seasonality, STL+LGBM is the pragmatic choice.

---

*Rutvik Prajapati*

# TFNSW + BOM - Weather and GTFS-R Data Analysis for Delay PRediction

## Data Pipeline & Modelling Reasoning

**Project:** NSW Public-Transport Weather-Resilience Predictor
**Scope of this document:** _JUST_ the notebook (the data collection/scrape server is at [Master Repo](https://github.com/qualiky/tfnsw-bom-data-analysis)

## 1. Project Overview & Research Question

**The question we are trying to answer:**
> *Given current weather conditions, the time of day, and basic context about a train or light-rail trip, how many seconds of delay should we expect — and what is the probability that the delay will exceed five minutes (a "significant" delay from the rider's point of view)?*

This is a **supervised regression + classification** problem:

| Target               | Type            | Unit           |
| -------------------- | --------------- | -------------- |
| `delay_seconds`      | Regression      | seconds        |
| `delay > 300s`       | Binary classify | 0 / 1          |

We are *not* forecasting the network state (that would be a time-series problem). We are estimating, for a given observation of `(weather, hour, line, ...)`, the conditional distribution of delay.

That framing — **conditional distribution, not forecast** — drove several choices later on (no ARIMA/Prophet, no recurrent models, yes gradient boosting).

---

## 2. The Data at a Glance

The training parquet contains **884,736 rows × 22 columns**, spanning **2026-03-31 → 2026-04-21** (≈ 22 days of continuous Sydney operations).

Key feature families:

- **Weather** (from BOM, nearest station per stop): `air_temp_c`, `rain_trace_mm`, `wind_gust_kmh` + a handful of dropped/sparse columns (see §3).
- **Temporal:** `hour`, `day_of_week`, derived `is_peak_hour`, `is_weekend`.
- **Trip context:** `line_group` (T1/T2/T4/T8/T9/L1–L4), `distance_to_terminus`, `occupancy_encoded`.
- **Signal flags:** `is_incident_active`, `is_ghost_train`.
- **Target:** `delay_seconds` (signed, in seconds).

This is a *wide-short* dataset compared to typical public-transport ML work (e.g. NYC taxi has years of history) — it is **critical** to the reader that **everything downstream must respect this limitation**. Twenty-two calendar days in a single Sydney autumn is a tiny weather sample.

---

## 3. Data Quality & Preprocessing

Raw GTFS-R is notoriously messy. Before any modelling we spent real effort *finding* and *removing* the noise that would otherwise dominate the loss.

### 3.1 Corrupt delay outliers

In the raw joined data the `delay_seconds` column ranged from **−76,387 s to +86,400 s** — i.e. 21-hour "delays" in both directions. These are overwhelmingly cases where a GTFS-R `arrival.delay` field was filled with clock-drift, a mislabelled trip, or a roll-over.

**Decision:** clip to `[0, 3600]` seconds.

| Choice | Why rejected / accepted |
| --- | --- |
| Drop all outliers outright | Too aggressive — loses legitimate 20-30 min delays during incidents |
| Winsorise to percentiles (e.g. 1st/99th) | Data-dependent — percentile on a zero-inflated distribution is unstable |
| Clip to signed `[-300, 3600]` (earlier version) | Allowed "early" running but made **Tweedie regression impossible** (Tweedie requires y ≥ 0) |
| **Clip to `[0, 3600]` (what we chose)** | Retains every legitimate delay, enables Tweedie, only ~30 rows (< 0.01 %) touched at the lower bound |

### 3.2 Dead columns

During EDA we observed several weather columns that were 100 % null in the collection window (humidity, pressure, dewpoint — BOM doesn't publish these consistently at the stations we snapped to). LightGBM will happily accept all-null columns, but every split it evaluates on them is wasted compute and adds slight variance. **All 100 %-null columns were dropped at preprocessing.**

### 3.3 Cancelled trips

Cancelled services appear in GTFS-R with `schedule_relationship = CANCELED` and carry no meaningful delay value. Including them would force the model to regress on an undefined target. **Cancelled rows are filtered out** before the train/test split.

### 3.4 Multi-shard parquet

The collector writes one parquet shard per day. We read them back with a **DuckDB glob pattern** (`data/parquet/*/part-*.parquet`). DuckDB's vectorised pushdown means we never materialise the full dataset in pandas during EDA — crucial for notebook responsiveness on a laptop.

### 3.5 Weather join method — ASOF

Weather observations are 30 min apart; trip events are ~ 15 s apart. A naïve equality join would drop 99 %+ of rows. We used a **DuckDB `ASOF LEFT JOIN`** that, for each trip event, pulls the *most recent previous* weather reading from the nearest BOM station. This guarantees **no future leakage** — a 14:00 weather reading can never be used for a 13:58 trip.

---

## 4. Exploratory Data Analysis (EDA)

EDA (`notebooks/01_eda.ipynb`) is where several modelling decisions were *forced* by the data. Four findings dominated:

### 4.1 Zero-inflation of the target

**~64.5 %** of all rows have `delay_seconds = 0`. A further ~20 % sit in the `(0, 60 s]` band. The remaining ~15 % is a long, heavy right tail up to the 3,600 s cap.

This is the single most important fact about the dataset. It dictated:

- Dropping L1/MAE regression objectives (they minimise to the **median**, which is zero — so the model literally predicts `0` for everything and "wins" MAE while scoring R² < 0).
- Choosing **Tweedie regression** (compound Poisson-Gamma — the canonical distribution for "lots of zeros, then a positive continuous tail").
- Splitting the problem into a **regressor + a separate classifier** rather than a single head.

### 4.2 Heavy tails + outliers *before* clipping

Before §3.1's clip, the min was **-76,387 s** and the max was **86,400 s**. These were confirmed by plotting the raw histogram on a log-y axis. This directly motivated the clip.

### 4.3 Narrow weather envelope

The collection window ran **2026-03-31 → 2026-04-21** (late-autumn Sydney). Temperatures observed: **14 °C – 27 °C**. Rainfall: mostly zero, a handful of light-rain days. No heatwaves (> 35 °C), no storm fronts, no frosts.

Sydney's **annual climate norms** span roughly 5 °C – 42 °C and include multi-day > 100 mm rain events. So the weather columns carry limited *useful* variance for a resilience model whose whole premise is "what happens in extreme weather?". We overlay the annual norms on the observed-range plots in the EDA notebook to make this clear.

### 4.4 `is_incident_active` is *near-constant* (not dead)

Early external feedback claimed this column was "all zeros from a broken alert-matching query". We verified: it is actually **98.3 % = 1**. Sydney's alert feed is noisy and nearly always contains *some* active alert somewhere on the network, and our join is spatially broad. So the feature isn't buggy — it just has **near-zero variance** (Bernoulli variance ≈ 0.017) and therefore low predictive power in this window. We kept it in the feature set (it is still informative in the rare 1.7 % of "quiet" moments) but noted it as a candidate for **redefinition**, not removal (see §14).

---

## 5. Feature Engineering

### 5.1 Spatial: Haversine distance for stop → weather-station mapping

Every GTFS stop was matched to its nearest BOM station via the **haversine great-circle formula** on (lat, lon) in degrees. We chose haversine over:

| Alternative | Why rejected |
| --- | --- |
| Euclidean on raw lat/lon | Wrong units, distorts near the poles, but also just *wrong* at Sydney's latitude (1° lon ≠ 1° lat) |
| Projected (EPSG:28356 GDA94 MGA56) + Euclidean | More accurate at < 1 m, but we only need ~ km-scale nearest-neighbour — overkill |
| Road-network distance (OSRM) | Correct transit-time proxy but **massive** dependency for a join that only needs "which BOM gauge is closest" |

Haversine is exact enough for a "pick one of ~ 25 stations within 10 km" problem and has no external dependencies.

### 5.2 Temporal: ASOF weather join, Sydney-local hour, DST-safe

- `hour` and `day_of_week` are computed in **`Australia/Sydney`** via `zoneinfo.ZoneInfo`, not UTC. A UTC hour would scramble peak-hour signal.
- `zoneinfo` (stdlib, Python 3.9+) was chosen over `pytz` because pytz's `localize()`/`normalize()` idiom is error-prone, and `zoneinfo` handles DST via IANA rules automatically.
- `is_peak_hour = hour ∈ {7,8,9} ∪ {16,17,18}` and `is_weekend = day_of_week ≥ 5` are **redundant** with `hour` and `day_of_week` for a tree model, but they are cheap, interpretable in SHAP, and survive because they help with partial-dependence plots.

### 5.3 Trip-context: `distance_to_terminus`

Computed per-event as "how many stops remain in this trip". It encodes the physical reality that delays **accumulate** along a trip — the last stop of a long route has more chance of being late than the first. This is a leakage-safe feature because it is known at prediction time.

### 5.4 Categorical: `line_group`

Ten values (T1, T2, T4, T8, T9, L1, L2, L3, L4, plus "OTHER"). We pass this as a **pandas `Categorical` with an explicit category list** so that LightGBM uses its built-in categorical-split algorithm (Fisher's exact ordering for categorical splits) rather than one-hot explosion.

| Alternative | Why rejected |
| --- | --- |
| One-hot encoding | 10 columns, LightGBM wastes splits on "is-not-T1" etc. |
| Target-mean encoding | Leakage risk unless fold-encoded; adds complexity |
| Frequency encoding | Throws away identity of low-frequency lines |
| **Native categorical (what we chose)** | No information loss, no leakage, LightGBM handles it natively |

The explicit category list matters at inference time: when the API receives a request for `line_group="T1"`, we must re-attach the same Categorical dtype with the *same ordering* the model saw during training — otherwise LightGBM will see it as an unseen level and route it down the missing-value branch. We persist the category list to `models/feature_meta.json` for this reason.

---

## 6. Train / Test Split Strategy

**Rule:** the test set is the *last 3 calendar days* of the window. No shuffling. No k-fold.

### Why not random 80/20?

Because time-series data has two kinds of leakage a random split can't prevent:

1. **Autocorrelation leakage:** two events 30 seconds apart on the same trip share enormous amounts of context. If one is in train and the other in test, the model's "test score" is really a "how well can I interpolate between two tightly correlated rows" score.
2. **Weather leakage:** a heatwave afternoon contributes rows to both train and test; the model memorises the afternoon's weather pattern and claims credit for "predicting" it.

A **temporal split** forces the model to generalise to *weather and operational conditions it has never seen before*, which is the deployment scenario.

### Why last 3 days and not last 1 day or last week?

- 1 day is too few rows (~ 40 k) and too easily dominated by a single operational event.
- 1 week would leave only ~ 15 days of training data.
- 3 days gives ~ 120 k test rows across at least one weekday and one weekend day — enough for stable metrics without starving training.

### What we rejected

| Alternative | Why rejected |
| --- | --- |
| Random 80/20 shuffle | Leakage (above) |
| K-Fold CV | Same leakage problem; tree models also don't need CV for mean performance estimates when you have 884 k rows |
| TimeSeriesSplit (rolling) | Correct but expensive; defers to future work when we have months of data |
| Leave-one-line-out | Would answer a different question ("can we generalise across lines?") — good follow-up study |

---

## 7. Model Selection — Delay Regressor

### 7.1 What we chose

**LightGBM with a Tweedie objective (variance power = 1.5)**, early-stopping on validation RMSE.

### 7.2 The journey — L1 → L2 → Tweedie

This was the single most educational part of the project.

**Attempt 1: L1 / MAE objective**
- Rationale: "MAE is robust to outliers."
- Result: validation R² = **−0.076** (worse than predicting the mean).
- Why it failed: L1 minimises at the **conditional median**. With 64.5 % zeros, the conditional median is **zero almost everywhere**. So the "optimal" L1 predictor is `0.0` for nearly every row. That beats MAE but tanks R².

**Attempt 2: L2 / MSE objective**
- Rationale: "Then let's minimise squared error directly."
- Result: R² slightly positive but model chased the heavy right tail, producing large predictions on benign inputs.
- Why it was suboptimal: L2 assumes Gaussian residuals; our residuals are a point-mass at 0 plus a Gamma-like tail. The normality assumption is violated violently.

**Attempt 3: Tweedie with `variance_power = 1.5` (what we chose)**
- Rationale: the Tweedie family of distributions is the canonical model for **non-negative data with a point-mass at zero and a positive-real tail** — exactly our shape.
  - `p = 1` → Poisson (no continuous tail)
  - `p = 2` → Gamma (no point-mass at zero)
  - `p ∈ (1, 2)` → **Compound Poisson-Gamma**: zeros from a Poisson count × Gamma severity. This is literally the insurance-claims / rainfall / ride-delays model.
  - We chose `1.5` as a common default; `1.3` and `1.7` were sanity-checked and gave near-identical validation scores.
- Result: R² = **+0.007**, ROC-AUC on the derived binary = 0.611. The sign flip (negative → positive) proves the fit now beats the mean baseline.

### 7.3 Why LightGBM over every alternative

| Model family | Why rejected |
| --- | --- |
| **Linear regression (OLS / Ridge / Lasso)** | Assumes linear relationship between `(temp, rain, hour, ...)` and delay. Our SHAP plots show strongly non-linear thresholds (e.g. delay only climbs above ~ 30 °C). OLS also has no native categorical, no built-in missing-value handling, and no Tweedie head in most standard libraries. |
| **GLM with Tweedie link (statsmodels)** | Would give us Tweedie but still linear-in-features — misses interactions. Also ~ 100× slower than LightGBM at this row count. |
| **Random Forest** | No Tweedie objective, no early stopping, much slower on 884 k × 22. Bagging = less efficient use of data than boosting. |
| **XGBoost** | Very close competitor; also supports Tweedie (`objective="reg:tweedie"`). We benchmarked: near-identical accuracy, LightGBM ~ 2× faster on this shape and has cleaner native categorical handling. |
| **CatBoost** | Best-in-class categorical handling, but slower training and no meaningful accuracy gain on a dataset with only one categorical column. |
| **Neural network (MLP / TabNet / FT-Transformer)** | Would need 10–100× more data to match gradient-boosting on tabular — see Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on tabular data?". Our 884 k rows / 12 features is squarely in GBDT territory. |
| **ARIMA / Prophet / seasonal decomposition** | These are *forecasting* models for aggregated time series. We are estimating a conditional distribution given features, not forecasting next-hour delay on one line. |
| **Quantile regression** | Genuinely interesting alternative (would give us delay percentiles). Parked as future work; see §14. |

### 7.4 Hyperparameters

We deliberately kept the LightGBM config **simple and uncranked**:

```
num_leaves        = 63
learning_rate     = 0.05
feature_fraction  = 0.9
bagging_fraction  = 0.8
bagging_freq      = 5
min_data_in_leaf  = 100
```

Justification: with only 22 days of weather variation, aggressive tuning (grid/Optuna) would overfit to this particular window. Simple defaults generalise better when the *data distribution itself* is the bottleneck.

---

## 8. Model Selection — Significant-Delay Classifier

### 8.1 What we chose

**LightGBM binary classifier** with `scale_pos_weight = 25`, predicting `delay_seconds > 300`.

### 8.2 Why a separate classifier at all?

Couldn't we just threshold the regressor's output at 300 s? In principle yes, but two reasons not to:

1. A regressor trained on Tweedie loss optimises mean-squared-log-ish error, which is not the same loss as binary cross-entropy for the threshold question. You get a better-calibrated probability by training the classification head directly.
2. The two heads can disagree, and that disagreement is *itself* a useful signal to the front-end ("model is uncertain").

### 8.3 The class-imbalance problem

Positive class (`delay > 300 s`) is **3.86 %** of rows. Negative is 96.14 %. Ratio ≈ 25 : 1.

Default LightGBM trained on this data produced probabilities clustered near 0.04 (just the base rate). Every threshold > 0.04 gave zero positive predictions. ROC-AUC looked fine (0.61) but F1 at the default 0.5 threshold was ~ 0.

### 8.4 Rebalancing: `scale_pos_weight` vs alternatives

| Technique | Why rejected / chosen |
| --- | --- |
| **`scale_pos_weight = 25` (what we chose)** | One line of config. Re-scales gradient contribution of positives by 25×, which is mathematically equivalent to duplicating positive rows. No data loss, no synthetic samples. |
| `is_unbalance = True` | LightGBM's "auto" version — it picks a ratio for you, but it picks `neg/pos` which is too aggressive when you already know your own cost structure. We moved off this to `scale_pos_weight=25` for explicit control. |
| Random under-sampling majority | Discards 90 %+ of training data. Wasteful with 884 k rows. |
| Random over-sampling minority | Duplicates exact rows — trees overfit the duplicates. |
| **SMOTE** | Synthesises minority examples by interpolating k-nearest-neighbour pairs. Genuinely useful for continuous features but **undefined for categoricals** (and for our integer `distance_to_terminus` it produces non-integer values). Also, SMOTE on a temporal dataset risks synthesising events that bridge train/test. |
| Focal loss | Great for object detection; overkill here and not natively in LightGBM. |
| Train a cost-sensitive threshold instead | We do this *as well* — see §9 — but it's complementary to, not a replacement for, reweighting. |

### 8.5 The common trap: "F1 = 0.14, the model is broken"

After rebalancing, F1 at the default 0.5 threshold was still ~ 0.14. This looks bad until you realise:

- Base-rate precision is 3.86 %.
- Random-guess recall at some threshold is 50 %.
- Random F1 ≈ 2 · 0.0386 · 0.5 / (0.0386 + 0.5) = **0.072**.

Our 0.14 is **~ 2× better than random** — modest, but real. The mistake would be comparing 0.14 to F1 benchmarks from balanced datasets.

---

## 9. Threshold Optimisation

After training, we do **not** use the default 0.5 probability threshold. Instead we sweep the precision-recall curve and pick the threshold that maximises F1 on the validation fold.

For this dataset that threshold landed at **0.54**, giving F1-optimal = **0.142**.

### Why this matters

With `scale_pos_weight = 25`, the raw probabilities are *not* calibrated to the true 3.86 % base rate — they're calibrated to an effective 50 % base rate. So a "probability" of 0.54 under this model corresponds roughly to a true P(delay > 300 s) of maybe 8 – 12 %. This is a design choice: we want the model to **say "likely delayed" more often than the base rate would suggest**, because the cost of missing a real delay (passenger frustration) is higher than the cost of a false alarm (mild user annoyance).

### Alternatives considered

| Approach | Why not (primary) |
| --- | --- |
| Fixed threshold = 0.5 | Arbitrary; wrong under class re-weighting |
| Youden's J (TPR - FPR max) | Optimises for balanced classes; we care about F1 at this base rate |
| Cost-matrix-driven threshold | Requires explicit (FN cost / FP cost) which we don't have from a product team |
| Platt / isotonic calibration then 0.5 | Clean theoretically; adds a second model to maintain. Equivalent in outcome to the F1 sweep for this pipeline. |

---

## 10. Evaluation Metrics

We report **five** numbers and refuse to collapse them to one:

| Metric | What it tells us |
| --- | --- |
| RMSE (seconds) | Regressor error in the natural unit the user cares about |
| MAE (seconds) | Regressor error, robust to the heavy tail |
| **R²** | Whether the regressor beats "predict the mean" at all |
| ROC-AUC | Ranking quality of the classifier, threshold-free |
| F1-optimal (+ threshold) | Operating-point quality after threshold sweep |

### Why R² is the star, not MAE

MAE was the first metric we reported. It looked great (~ 45 s). But a constant-zero predictor *also* gets a great MAE here because of §4.1. R² is the only metric of the five that can expose "your model is worse than a dummy", which is *exactly* what happened with the L1 objective. We now treat R² as the go/no-go gate.

### Why we don't report MAPE

`delay_seconds` has a point-mass at zero. `abs(y - ŷ) / y` is undefined for 64.5 % of rows. MAPE is unusable here.

### Why we don't report R² alone

R² is scale-free and doesn't tell the rider "your train will be X seconds late". RMSE and MAE put the error in human units.

---

## 11. Feature Importance with SHAP

We compute **SHAP values** via `shap.TreeExplainer` on both trained models.

### Why SHAP and not LightGBM's built-in importance?

LightGBM exposes three importance types (`split`, `gain`, `permutation`). All three suffer from a well-known bias: high-cardinality continuous features (`air_temp_c`, `distance_to_terminus`) dominate because they offer more split points. This would *always* put `air_temp_c` above `is_ghost_train` regardless of actual predictive contribution.

SHAP (Shapley values from cooperative game theory) attributes each prediction's deviation from the mean to each feature *additively*, averaging across coalitions. It is:

1. **Locally faithful** — the attributions sum to the model's output.
2. **Globally consistent** — if a feature's importance changes, its SHAP importance moves the same direction.
3. **Cardinality-fair** — does not inflate continuous features.

The trade-off is compute: SHAP on 884 k rows takes minutes, vs. milliseconds for gain-based importance. For a one-off training run, that's acceptable.

### Alternatives considered

| Method | Why rejected |
| --- | --- |
| `feature_importance(importance_type="gain")` | Cardinality bias |
| Permutation importance (`sklearn.inspection`) | Breaks feature correlations — attributes correlated features unfairly |
| LIME | Local only, high variance, not stable across runs |
| Mean decrease in impurity | Same cardinality bias as gain |

---

## 12. Contextualising the Results

Final numbers on the held-out last-3-days test set:

| Metric | Value | Interpretation |
| --- | --- | --- |
| R² | **+0.007** | Marginally beats the mean predictor. Weather adds ~ 0.7 % of explained variance **over the baseline** in this window. |
| ROC-AUC | **0.611** | Noticeably better than random (0.5). Model ranks delayed-vs-on-time correctly ~ 61 % of the time. |
| F1-optimal | **0.142** at threshold 0.54 | ~ 2× the random-guess F1 at this base rate. |

### What this number is *not*

- Not a failure of the model architecture. We verified this by computing baseline R²s for the mean predictor (−0.0007) and the median predictor (−0.086). Our +0.007 is on the right side of zero.
- Not a failure of feature engineering — SHAP shows `hour`, `line_group`, and `air_temp_c` all contributing as expected.

### What this number *is*

- **A direct consequence of the 22-day window.** In April 2026 Sydney, temperatures ran 14 – 27 °C. The model literally never saw a heatwave, a storm, or a frost — the exact conditions where weather *does* drive delays. The model can only learn what the data shows it.
- Evidence that on **benign weather days**, weather explains very little of the delay variance — most delay comes from operational factors (incidents, crowding, signalling) that aren't in this feature set.

This is the honest story: **the ceiling on R² here is set by the data, not the model**. With more data (§13) and more features (§14), the same architecture will have room to grow.

---

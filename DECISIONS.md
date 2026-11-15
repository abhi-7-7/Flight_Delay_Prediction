# Phase 2 Decision Log

This file records how the Phase 2 (deep learning) design was reached.
There are two kinds of decision:

- **Design rules:** constraints fixed by reasoning before any experiment. They define the problem and are never tuned.
- **Experimental choices:** settled by comparing candidates on the **validation split (Sep–Oct 2013)**. The test split (Nov–Dec 2013) is used **once**, in Stage 8, on the final model only.

Every run is appended to [`experiments/results.csv`](experiments/results.csv), including runs that lost. Each run's exact settings are in `configs/`.

---

## Part A: Design rules (fixed)

| # | Rule | Reason |
|---|---|---|
| R1 | **Prediction cutoff: 6:00 PM local time on the day before departure (D−1).** | Same as Phase 1, so Phase 2 results are directly comparable. Gives a usable advance warning. |
| R2 | **Target: arrival delay ≥ 15 minutes.** | US DOT on-time definition, used in Phase 1 and the literature. |
| R3 | **Chronological splits:** train Jan–Aug, validation Sep–Oct, test Nov–Dec. | Mimics predicting future flights. Same as Phase 1. |
| R4 | **No post-departure inputs** (`dep_time`, `dep_delay`, `arr_time`, `air_time`, `arr_delay`, `tailnum`, …). | They are unknown at the cutoff (leakage). |
| R5 | **A past flight's outcome counts as known only once it has actually landed**, using its actual arrival converted to New York time. For flights on day D, history may only use outcomes known before 6 PM on D−1. | Strict version of R1 for history features. |
| R6 | **"Observation days" run 6 PM → 6 PM.** Observation day *k* covers 18:00 on *k−1* to 17:59 on *k*. A flight on D uses observation days D−14 … D−1, all fully known at the cutoff. | Every history step has the same length and is complete; no partial days. |
| R7 | **Weather for main models: only hourly observations up to 17:xx on D−1**, summarised per observation day and origin. | Departure-time weather is not known at the cutoff. |
| R8 | **One "perfect forecast" experiment** may use actual weather in a ±2 h window around scheduled departure. It is always labelled as an **upper bound**, never as the headline result. | Measures how much a good weather forecast would be worth, without breaking R1. |
| R9 | **Join weather on `origin, year, month, day, hour`**, never on `time_hour` (UTC in `weather.csv`, local time in `flights.csv`). | Avoids a silent 4–5 hour shift. |
| R10 | **Model selection on validation AP; threshold by maximum validation F1.** | Same protocol as Phase 1. |
| R11 | **Deep learning configurations are run with 3 seeds**; mean ± std is reported. | Neural networks vary from run to run; one run cannot show that one option beats another. |

### Data sources

| File | Source | Use |
|---|---|---|
| `Datasets/flights.zip` | NYC 2013 flights (Phase 1) | Flights, labels, schedule |
| `Datasets/weather.csv` | `tidyverse/nycflights13` `data-raw/weather.csv` (Iowa Environmental Mesonet ASOS) | Hourly origin weather |
| `Datasets/airports.csv` | `tidyverse/nycflights13` `data-raw/airports.csv` | Destination time zones (needed for R5) |

---

## Part B: Experimental choices

Status key: ⏳ pending · ✅ decided

| Stage | Question | Candidates | Status |
|---|---|---|---|
| 0 | What must the deep model beat? | Phase 1 LR / XGBoost; + hand-made lag features | ✅ |
| 1 | Which history branches help? | none / origin / route / origin + route | ✅ |
| 2 | How far back? | 7 vs 14 days | ⏳ |
| 3 | How much is weather worth? | none / daily (≤ cutoff) / + perfect forecast (upper bound) | ⏳ |
| 4 | Which sequence encoder? | GRU vs 1D CNN | ⏳ |
| 5 | How to encode categories? | one-hot vs embeddings; ± flight-number embedding | ⏳ |
| 6 | How much regularisation? | dropout, weight decay, early stopping | ⏳ |
| 7 | Which hybrid? | A: NN representation → XGBoost/LR · B: stacking · C: residual; for A: out-of-fold vs split-train vs naive | ⏳ |
| 8 | Final test (run once) | the selected configuration | ⏳ |

Each stage below is filled in from `experiments/results.csv` once its runs are complete.

### Stage 0: Baselines

**Question:** What does the deep model have to beat? Can simple hand-made history features already capture what a sequence model would learn?

**Runs:** `configs/stage0/*.yaml`. XGBoost uses 3 seeds. Logistic Regression (lbfgs) is deterministic, so it is run once.

**Validation results (Sep–Oct 2013):**

| Experiment | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 |
|---|---|---:|---:|---:|---:|---:|
| dummy_prior | dummy | 1 | 0.1570 ± 0.0000 | 1.00× | 0.5000 | 0.2713 |
| phase1_features | logreg | 1 | 0.2349 ± 0.0000 | 1.50× | 0.6449 | 0.3267 |
| phase1_plus_lags | logreg | 1 | 0.2350 ± 0.0000 | 1.50× | 0.6462 | 0.3289 |
| phase1_features | xgboost | 3 | 0.2336 ± 0.0160 | 1.49× | 0.6339 | 0.3206 |
| phase1_plus_lags | xgboost | 3 | 0.2413 ± 0.0018 | 1.54× | 0.6494 | 0.3304 |

**Findings:**

1. **Phase 1 is reproduced.** Logistic Regression gives validation AP 0.2349 / ROC-AUC 0.6449, against 0.2348 / 0.6446 reported in Phase 1. The Phase 2 pipeline therefore starts from the same point.
2. **Phase 1's XGBoost result depends on the seed.** Across 3 seeds, validation AP ranges from 0.2216 to 0.2518 (std 0.016). The single-seed Phase 1 comparison (LR 0.2348 vs XGBoost 0.2301) falls inside that noise, so "LR beat XGBoost" was not a reliable conclusion. This supports rule R11 (multiple seeds).
3. **Hand-made lag features help XGBoost a little and LR not at all.** XGBoost improves from 0.2336 to 0.2413 mean AP, and its seed spread drops tenfold (0.016 → 0.0018). LR is unchanged, which suggests the history signal is non-linear (e.g. it matters mainly during disruption).
4. **The gain from history is modest.** Under the 6 PM D−1 cutoff, recent delay history adds about +0.008 AP. Any deep sequence model has to beat this to justify itself.

**Decision:** The target to beat is **XGBoost + hand-made lags (val AP 0.2413 ± 0.0018)**. Every later stage is compared against this row, not against Phase 1.

### Stage 1: Which history branches help?

**Question:** Does the network gain from reading recent history as a sequence, and from which level (origin airport, route, or both)?

**Setup:** `configs/stage1/*.yaml`. Same network for all four runs (static branch with embeddings for carrier / origin / grouped destination + Phase 1 numerics; GRU encoders, hidden size 32; dropout 0.2; AdamW, weight decay 1e-4). 14-day window, no weather, no flight-number embedding. 3 seeds each. Early stopping on validation AP (patience 4).

**Validation results (Sep–Oct 2013):**

| Experiment | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 |
|---|---|---:|---:|---:|---:|---:|
| static_only | nn_gru | 3 | 0.2394 ± 0.0052 | 1.53× | 0.6456 | 0.3272 |
| origin_only | nn_gru | 3 | 0.2586 ± 0.0015 | 1.65× | 0.6662 | 0.3426 |
| route_only | nn_gru | 3 | 0.2522 ± 0.0027 | 1.61× | 0.6585 | 0.3356 |
| origin_route | nn_gru | 3 | **0.2645 ± 0.0066** | 1.69× | 0.6745 | 0.3507 |
| _Stage 0 target: XGBoost + lags_ | xgboost | 3 | 0.2413 ± 0.0018 | 1.54× | 0.6494 | 0.3304 |

**Check for early-stopping bias.** The networks choose their best epoch on validation, which the tabular baselines do not. Validation AP at a **fixed** epoch (mean of 3 seeds, from `artifacts/curves/`):

| Experiment | Epoch 1 | Epoch 3 | Epoch 5 |
|---|---:|---:|---:|
| static_only | 0.2369 | 0.2315 | 0.2318 |
| origin_only | 0.2476 | 0.2538 | 0.2383 |
| route_only | 0.2403 | 0.2400 | 0.2452 |
| origin_route | 0.2542 | 0.2610 | 0.2429 |

**Findings:**

1. **The network on its own adds nothing.** With no history, the network (0.2394) matches XGBoost + lags (0.2413). Embeddings and dense layers on the Phase 1 features are not where the gain comes from.
2. **Learned history clearly helps.** Origin history adds about +0.019 AP and route history about +0.013. Together they reach **0.2645**, roughly +0.025 over the static network and +0.023 over XGBoost + lags. The same hand-made history summaries gave XGBoost only +0.008 (Stage 0), so the sequence encoders extract more from the same data.
3. **The two branches complement each other.** Origin history (airport-wide disruption) and route history (route and destination conditions) each help separately, and more together.
4. **The gain is not an early-stopping artefact.** At a fixed epoch 3, origin + route still beats static only by about 0.03. Early stopping adds a little (0.2610 → 0.2645) but does not change the ranking.
5. **The networks overfit quickly.** Validation AP peaks at epoch 2–3 and falls by epoch 5, even though training loss keeps improving. The likely cause is the shift between training months (26% delayed) and validation months (16% delayed). This feeds directly into Stage 6 (regularisation).

**Decision:** Keep **both history branches (origin + route)** in the model. New best: **val AP 0.2645 ± 0.0066**.

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
| 2 | How far back? | 7 vs 14 days (+ 1, 3, 5 added to bracket the best) | ✅ |
| 3 | How much is weather worth? | none / daily (≤ cutoff) / + perfect forecast (upper bound); + 2 timing controls | ✅ |
| 4 | Which sequence encoder? | GRU vs 1D CNN (pooled / last-day readout) vs flatten control; 3 vs 7 days | ✅ |
| 5 | How to encode categories? | one-hot vs embeddings; ± flight-number embedding | ✅ |
| 6 | How much regularisation? | dropout, weight decay, early stopping; + learning rate, GRU size, 3 vs 5 days | ✅ |
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

### Stage 2: How far back?

**Question:** How many observation days of history should each branch read? The plan was 7 vs 14 days. After the first results showed shorter was better, 1, 3 and 5 days were added to find where the gain stops.

**Setup:** `configs/stage2/*.yaml`. Same network as the Stage 1 winner (origin + route GRU branches, no weather). Only the window length changes. 3 seeds each. The 14-day run repeats Stage 1's `origin_route` and gives exactly the same numbers (0.2645 ± 0.0066), which confirms that runs are reproducible.

**Validation results (Sep–Oct 2013):**

| Window | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 |
|---|---|---:|---:|---:|---:|---:|
| 1 day | nn_gru | 3 | 0.2533 ± 0.0016 | 1.61× | 0.6614 | 0.3369 |
| 3 days | nn_gru | 3 | **0.2728 ± 0.0092** | 1.74× | 0.6786 | 0.3528 |
| 5 days | nn_gru | 3 | 0.2677 ± 0.0022 | 1.71× | 0.6760 | 0.3514 |
| 7 days | nn_gru | 3 | 0.2655 ± 0.0048 | 1.69× | 0.6731 | 0.3492 |
| 14 days | nn_gru | 3 | 0.2645 ± 0.0066 | 1.69× | 0.6745 | 0.3507 |

**Same-seed comparison against 7 days** (val AP difference for seeds 42 / 43 / 44):

| Window | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| 1 day | −0.0103 | −0.0159 | −0.0105 |
| 3 days | +0.0033 | +0.0123 | +0.0062 |
| 5 days | +0.0023 | −0.0016 | +0.0057 |
| 14 days | +0.0004 | +0.0005 | −0.0041 |

**Fixed-epoch check** (mean val AP of 3 seeds, from `artifacts/curves/`):

| Window | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 5 |
|---|---:|---:|---:|---:|
| 1 day | 0.2441 | 0.2433 | 0.2433 | 0.2445 |
| 3 days | 0.2584 | 0.2645 | 0.2701 | 0.2635 |
| 5 days | 0.2539 | 0.2651 | 0.2677 | 0.2457 |
| 7 days | 0.2547 | 0.2641 | 0.2635 | 0.2514 |
| 14 days | 0.2542 | 0.2623 | 0.2610 | 0.2429 |

**Findings:**

1. **Some sequence is needed.** Using only yesterday (1 day) is clearly worse than 3 days (−0.020 AP, worse in every seed). So the model gains from seeing how the last few days developed, not just from the latest day.
2. **Beyond about 3 days, older history does not help.** 7 and 14 days are indistinguishable (paired differences of ±0.004). 3 days beats 7 days in all three seeds (+0.003 to +0.012). Disruption at NYC airports is short-lived, so the most recent few days carry the signal.
3. **Longer windows overfit faster.** At epoch 5, the 14-day model has fallen to 0.2429, while the 3-day model is still at 0.2635. More inputs per flight give the network more ways to fit patterns specific to the training months (see Stage 1 finding 5).
4. **The ranking holds without early stopping.** At a fixed epoch 3, 3 days (0.2701) is still the best window.
5. **Caveat:** the 3-day window has the largest seed spread (± 0.0092; seed 43 reaches 0.2833). Its lead over 5 days is small and partly within noise. Stage 6 (regularisation) should re-check 3 vs 5 days if it changes how fast the model overfits.

**Decision:** Use a **3-day window** for both branches from Stage 3 on. New best: **val AP 0.2728 ± 0.0092**.

### Stage 3: How much is weather worth?

**Question:** Does origin-airport weather known at the cutoff improve the model? And how much more would a perfect forecast of departure-time weather add?

**Setup:** `configs/stage3/*.yaml`. Stage 2 winner (origin + route GRU, 3-day window), 3 seeds each.

- `no_weather`: same setup as Stage 2's 3-day run.
- `daily_weather`: the 15 daily summaries (rule R7, hours up to 17:xx on each observation day) are added to every step of the **origin** sequence, with a `weather_obs` flag for days with no reports.
- `daily_plus_forecast`: also adds the 7 `pf_*` columns (actual weather ±2 h around scheduled departure) to the static inputs. **Upper bound only (rule R8):** this information is not available at 6 PM on D−1.

**Validation results (Sep–Oct 2013):**

| Experiment | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 |
|---|---|---:|---:|---:|---:|---:|
| no_weather | nn_gru | 3 | 0.2728 ± 0.0092 | 1.74× | 0.6786 | 0.3528 |
| daily_weather | nn_gru | 3 | **0.3238 ± 0.0078** | 2.06× | 0.6867 | 0.3628 |
| _daily_plus_forecast (upper bound)_ | nn_gru | 3 | _0.3674 ± 0.0061_ | 2.34× | 0.7008 | 0.3824 |

Daily weather improves AP in every seed (+0.067 / +0.037 / +0.050 for seeds 42 / 43 / 44).

**Checking the daily-weather gain.** The gain (+0.051 AP) was larger than all the history gains combined, so it was checked before being accepted.

1. *Where it comes from.* Using the mean prediction of the 3 seeds: the correlation between a day's predicted and actual delay rate rises from 0.35 to 0.59. With the 5 worst validation days removed (12 Sep, 7 Oct, 2 Sep, 11 Oct, 13 Sep), the gain falls from +0.051 to +0.007 AP. Within a single day, ranking does not improve (mean per-day AP change −0.004). **Weather helps the model recognise bad days; it does not help it decide which flights on a given day are delayed.**
2. *Timing controls* (`configs/stage3/control_*.yaml`, new `weather_shift` option):

| Control | What the last window step shows | Val AP (mean ± std) | Val ROC-AUC |
|---|---|---:|---:|
| control_weather_one_day_older | obs-day D−2 (allowed, one day staler) | 0.2803 ± 0.0140 | 0.6542 |
| daily_weather (main run) | obs-day D−1 (up to 17:59 on D−1) | 0.3238 ± 0.0078 | 0.6867 |
| control_weather_same_day_leak | obs-day D (**leak**, not allowed) | 0.3828 ± 0.0053 | 0.7045 |

The main run falls between the two controls, as it should. Using day D's weather scores clearly higher (+0.059), so the main run is not seeing day D. Moving the weather one day back removes almost all the gain, so the useful signal is the **most recent** weather, ending just before the cutoff. This fits weather systems that last into the next day (rain or fog on the evening of D−1 continuing into D).

**Fixed-epoch check** (mean val AP of 3 seeds):

| Experiment | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 5 |
|---|---:|---:|---:|---:|
| no_weather | 0.2584 | 0.2645 | 0.2701 | 0.2635 |
| daily_weather | 0.3180 | 0.3228 | 0.2986 | 0.2759 |
| daily_plus_forecast | 0.3509 | 0.3674 | 0.3548 | 0.3314 |

**Findings:**

1. **Weather known at the cutoff is the largest single gain so far:** +0.051 AP over no weather and +0.083 over the Stage 0 target. It already helps at epoch 1, so it is not an early-stopping effect.
2. **The gain is episodic.** It comes mostly from a handful of disrupted days. This is a risk for the test split: if November–December has different weather events (e.g. snow, which the validation months never had), the gain may be smaller or larger. Stage 8 should report results per day as well as overall.
3. **A perfect forecast would be worth about +0.044 AP more** (0.3674). This is the value of good departure-time weather information. It is reported as an upper bound only and is **not** used in later stages.
4. **The weather model overfits even faster** (best at epoch 1–2, down to 0.2759 by epoch 5). Another input for Stage 6.

**Decision:** Keep **daily weather (≤ cutoff) in the origin branch**. Perfect forecast stays an upper-bound result only. New best: **val AP 0.3238 ± 0.0078**.

### Stage 4: Which sequence encoder?

**Question:** Which encoder should read the history windows: a GRU (recurrent) or a 1D CNN (convolution over days)? And does modelling the order of days matter at all?

**Setup:** `configs/stage4/*.yaml`. Stage 3 winner (origin + route branches, daily weather), 3 seeds each. Encoders:

- `gru`: GRU, hidden size 32, final hidden state.
- `cnn`: two Conv1d layers (kernel 3, 32 channels), then mean- and max-pooling over days.
- `flat` (control): flatten the window and apply one dense layer. It sees every day's values but has no built-in notion of order or recency.
- `cnn_last` (added after the first results): same CNN, but reads the output at the last day (D−1) instead of pooling.

GRU and pooled CNN were also run with a 7-day window, because Stage 2 chose 3 days with the GRU and no weather. A CNN might have used a longer window better.

**Validation results (Sep–Oct 2013):**

| Experiment | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 | Params | Train time / seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gru_3d | nn_gru | 3 | **0.3238 ± 0.0078** | 2.06× | 0.6867 | 0.3628 | 21,465 | 6 s |
| gru_7d | nn_gru | 3 | 0.3110 ± 0.0068 | 1.98× | 0.6766 | 0.3528 | 21,465 | 8 s |
| cnn_3d | nn_cnn | 3 | 0.2852 ± 0.0093 | 1.82× | 0.6814 | 0.3543 | 25,305 | 69 s |
| cnn_7d | nn_cnn | 3 | 0.2701 ± 0.0142 | 1.72× | 0.6648 | 0.3437 | 25,305 | 87 s |
| cnn_last_3d | nn_cnn_last | 3 | 0.3147 ± 0.0170 | 2.00× | 0.6888 | 0.3625 | 21,209 | 77 s |
| flat_3d (control) | nn_flat | 3 | 0.2969 ± 0.0122 | 1.89× | 0.6829 | 0.3577 | 15,001 | 2 s |

**Same-seed comparison against gru_3d** (val AP difference, seeds 42 / 43 / 44):

| Experiment | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| gru_7d | −0.0148 | −0.0159 | −0.0079 |
| cnn_3d | −0.0569 | −0.0258 | −0.0333 |
| cnn_7d | −0.0668 | −0.0345 | −0.0599 |
| cnn_last_3d | −0.0114 | −0.0250 | +0.0089 |
| flat_3d | −0.0498 | −0.0188 | −0.0123 |

**Fixed-epoch check** (mean val AP of 3 seeds):

| Experiment | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 5 |
|---|---:|---:|---:|---:|
| gru_3d | 0.3180 | 0.3228 | 0.2986 | 0.2759 |
| gru_7d | 0.3106 | 0.3092 | 0.2896 | 0.2752 |
| cnn_3d | 0.2852 | 0.2685 | 0.2649 | 0.2554 |
| cnn_7d | 0.2699 | 0.2627 | 0.2607 | 0.2529 |
| flat_3d | 0.2967 | 0.2912 | 0.2859 | 0.2816 |

**Findings:**

1. **The GRU is the best encoder.** It has the highest mean AP and beats the pooled CNN and the flatten control in every seed. It is also the fastest of the sequence encoders on CPU (about 6 s per run vs 70–90 s for the CNNs).
2. **Order matters.** The GRU beats the order-blind flatten control by +0.027 AP. Reading the days as a sequence is worth more than just seeing all their values.
3. **The pooled CNN loses because pooling throws away recency.** It scored even below the flatten control. Mean/max pooling over days forgets *which* day a pattern came from, but Stage 3 showed that the most recent day (D−1) carries the weather signal. Reading the CNN at the last day instead of pooling raises it from 0.2852 to 0.3147, which recovers most of the gap. This variant was added *after* the first results to test this explanation.
4. **The CNN with last-day readout is close to the GRU but less reliable.** Its seed spread is about twice the GRU's (± 0.017). It wins one seed and loses two, and it trains about 10× slower here. There is no reason to prefer it.
5. **3 days is still right with weather.** The GRU at 7 days is worse in every seed (−0.013 mean). This confirms Stage 2's choice now that weather is included.
6. **Every model peaks at epoch 1–2.** Fast overfitting does not depend on the encoder, so Stage 6 is the place to deal with it.

**Decision:** Keep the **GRU encoder** with a **3-day window**. Best remains **val AP 0.3238 ± 0.0078**.

### Stage 5: How to encode categories?

**Question:** Should carrier, origin and grouped destination enter the network as learned embeddings or as one-hot vectors? Does adding the flight number (about 3,800 values) help?

**Setup:** `configs/stage5/*.yaml`. Stage 4 winner (GRU, 3-day window, daily weather), 3 seeds each. One-hot is a frozen identity embedding, so only the encoding of the categories changes. Rare categories (fewer than 20 training flights) share one "rare" slot in both encodings, as in Phase 1.

**Validation results (Sep–Oct 2013):**

| Experiment | Model | Runs | Val AP (mean ± std) | AP lift | Val ROC-AUC | Val F1 | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| embeddings | nn_gru | 3 | **0.3238 ± 0.0078** | 2.06× | 0.6867 | 0.3628 | 21,465 |
| one_hot | nn_gru | 3 | 0.3150 ± 0.0112 | 2.01× | 0.6857 | 0.3629 | 34,330 |
| embeddings_flight | nn_gru | 3 | 0.3184 ± 0.0118 | 2.03× | 0.6886 | 0.3627 | 53,657 |
| one_hot_flight | nn_gru | 3 | 0.3071 ± 0.0091 | 1.96× | 0.6829 | 0.3606 | 3,953,706 |

**Same-seed difference against embeddings** (seeds 42 / 43 / 44): one_hot −0.0054 / −0.0082 / −0.0128; embeddings_flight −0.0008 / −0.0085 / −0.0070; one_hot_flight −0.0172 / −0.0119 / −0.0209.

**Findings:**

1. **Embeddings beat one-hot in every seed** (+0.009 AP on average) with fewer parameters. Small dense vectors let similar carriers and airports share what they learn.
2. **The flight number does not help.** It adds 32k parameters (embedding) or 3.9M (one-hot) and lowers AP in every seed. Most flight numbers have few training flights, and what they do carry (carrier, route, time of day) is already in the model.
3. The differences are small next to Stages 1–4. Category encoding is a minor choice for this problem.

**Decision:** Keep **learned embeddings** for carrier, origin and grouped destination, and **no flight-number input**. Best remains **val AP 0.3238 ± 0.0078**.

### Stage 6: How much regularisation?

**Question:** Every model so far peaks at epoch 1–2 and then gets worse. Which regularisation (dropout, weight decay, smaller model, smaller learning rate, early stopping) controls this best?

**Setup:** `configs/stage6/*.yaml`. Stage 5 winner as `reference` (dropout 0.2, weight decay 1e-4, learning rate 1e-3, GRU hidden 32, early stopping with patience 4). Each other config changes **one** setting. `no_early_stop` trains 10 fixed epochs and keeps the last one. (Its first run failed to start because of a YAML quoting error in the config. It was fixed in a separate commit and re-run; no results were lost or repeated.)

**Validation results (Sep–Oct 2013):**

| Experiment | Change | Runs | Val AP (mean ± std) | Val ROC-AUC | Val F1 | Best epochs (3 seeds) |
|---|---|---:|---:|---:|---:|---|
| reference | – | 3 | 0.3238 ± 0.0078 | 0.6867 | 0.3628 | 2, 2, 1 |
| dropout_04 | dropout 0.4 | 3 | 0.3208 ± 0.0070 | 0.6856 | 0.3624 | 2, 2, 1 |
| dropout_06 | dropout 0.6 | 3 | 0.3171 ± 0.0116 | 0.6826 | 0.3607 | 2, 2, 2 |
| wd_1e-2 | weight decay 1e-2 | 3 | 0.3240 ± 0.0074 | 0.6868 | 0.3630 | 2, 2, 1 |
| wd_1e-1 | weight decay 1e-1 | 3 | 0.3241 ± 0.0077 | 0.6868 | 0.3632 | 2, 2, 1 |
| hidden_16 | GRU hidden 16 | 3 | 0.3164 ± 0.0058 | 0.6870 | 0.3650 | 2, 2, 3 |
| **lr_3e-4** | learning rate 3e-4 | 3 | **0.3262 ± 0.0058** | 0.6865 | 0.3645 | 5, 4, 4 |
| no_early_stop | 10 epochs, keep last | 3 | 0.2595 ± 0.0163 | 0.6520 | 0.3348 | – |
| lr_3e-4_window_5d | lr 3e-4 and 5-day window | 3 | 0.3181 ± 0.0035 | – | – | 3, 4, 4 |

**Same-seed difference against reference** (seeds 42 / 43 / 44): no_early_stop −0.063 / −0.053 / −0.078; lr_3e-4 −0.0004 / +0.0004 / +0.0069; dropout and weight decay all within ±0.015 with no consistent sign. 5 vs 3 days at lr 3e-4: −0.015 / −0.006 / −0.003.

**Findings:**

1. **Early stopping is the regulariser that matters.** Without it, AP falls by 0.064 and the seed spread doubles. Dropout, weight decay and a smaller GRU change AP by less than the seed noise.
2. **Why the usual regularisers do not help.** The network is not overfitting in the classic sense (memorising noise). Training months have 26% delayed flights and validation months 16%, and patterns of the summer delay season stop applying in autumn. Stronger penalties cannot fix a change in the data. Stopping before the model fits the summer too closely can.
3. **A smaller learning rate is slightly better and more stable.** At lr 3e-4 the best epoch moves from 1–2 to 4–5. The model then depends less on one lucky epoch, and the seed spread falls (± 0.0058 vs ± 0.0078). The AP gain (+0.002) is within noise. It is adopted for stability, not for the AP.
4. **3 days is confirmed again.** At the new learning rate, 5 days is worse in every seed. This settles the caveat left in Stage 2.

**Decision:** Learning rate **3e-4**, early stopping (patience 4) on validation AP, dropout 0.2, weight decay 1e-4, GRU hidden 32, 3-day window. Best: **val AP 0.3262 ± 0.0058**. This is the final deep-learning model; Stage 7 tests whether hybrids can improve on it.

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
| 0 | What must the deep model beat? | Phase 1 LR / XGBoost; + hand-made lag features | ⏳ |
| 1 | Which history branches help? | none / origin / route / origin + route | ⏳ |
| 2 | How far back? | 7 vs 14 days | ⏳ |
| 3 | How much is weather worth? | none / daily (≤ cutoff) / + perfect forecast (upper bound) | ⏳ |
| 4 | Which sequence encoder? | GRU vs 1D CNN | ⏳ |
| 5 | How to encode categories? | one-hot vs embeddings; ± flight-number embedding | ⏳ |
| 6 | How much regularisation? | dropout, weight decay, early stopping | ⏳ |
| 7 | Which hybrid? | A: NN representation → XGBoost/LR · B: stacking · C: residual; for A: out-of-fold vs split-train vs naive | ⏳ |
| 8 | Final test (run once) | the selected configuration | ⏳ |

Each stage below is filled in from `experiments/results.csv` once its runs are complete.

### Stage 0: Baselines
_Pending._

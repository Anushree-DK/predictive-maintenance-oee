# Predictive Maintenance & OEE Command Center

Built for the [Snowflake CoCo CLI Hackathon 2026 — GCC Edition](https://hack2skill.com/event/cococlihack-gccedition/),
problem statement 3: **"Predictive Maintenance and OEE Command Center."**

**The problem**: manufacturers lose value to unplanned downtime because OT
sensor data sits apart from ERP and maintenance context. This converges IT
data (maintenance records, production/OEE) and OT data (turbofan sensor
telemetry) in Snowflake, predicts failures before they happen, and automates
the maintenance response — closing the loop from raw sensor reading to a
work order, with the outcome logged back for future learning.

End-to-end flow: sensor + maintenance + production data → Snowflake →
feature engineering + OEE (Snowpark SQL) → ML prediction (RUL, failure
probability, risk, confidence) → AI decision layer (Cortex) → Unified
Command Center dashboard with agentic actions → outcome logging that feeds
back into Snowflake. See [DATA_SOURCES.md](DATA_SOURCES.md) for exactly
which parts are real NASA sensor data vs. synthesized enterprise context.

## Stack

| Layer | Technology |
|---|---|
| Data | NASA C-MAPSS + synthetic maintenance/production data |
| Data platform | Snowflake |
| ML | Python + scikit-learn HistGradientBoostingRegressor (bagged ensemble for RUL + confidence) — swapped from XGBoost/LightGBM, both of which need Homebrew's `libomp` and weren't loadable in this environment |
| Feature engineering | Python (pandas) in mock mode, Snowflake SQL window functions in Snowflake mode |
| AI reasoning | Snowflake Cortex, templated fallback in mock mode or when Cortex is unavailable (e.g. trial-tier accounts) |
| Backend | none yet — Streamlit reads `src/` directly; add FastAPI only if a non-Streamlit client needs the API |
| Frontend | Streamlit |
| Visualization | Plotly |
| Agentic action | Python → `ACTION_OUTCOMES` (SQL write in Snowflake mode) |
| Business impact | Python — predictions → estimated $ downtime cost avoided (`src/business_impact.py`) |
| NL analytics | Cortex Analyst over a Semantic View (`src/cortex_analyst.py`) — built, not yet live-tested |
| Testing | pytest — 22 unit tests over OEE, feature engineering, ML helpers, business impact |

## Status

`SNOWFLAKE_MODE` in `.env` switches the whole pipeline between two backends
without changing any pipeline code (see `src/data_access.py`):
- `mock` — local `data/mock/*.csv` files, no Snowflake account needed.
- `snowflake` — real Snowflake tables + SQL feature engineering + Cortex.
  Currently set to `snowflake` and verified working end-to-end against a
  real trial account (see "Switching to real Snowflake" below).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # SNOWFLAKE_MODE=mock by default
```

## Run it (mock mode)

```bash
python scripts/generate_mock_data.py   # writes data/mock/*.csv (fully synthetic)
python -m src.ml.train                 # trains models/rul_model.joblib
streamlit run dashboard/app.py         # opens the Unified Command Center
```

### Using the real NASA C-MAPSS dataset instead of synthetic data

1. Download "Turbofan Engine Degradation Simulation Data Set" from NASA's
   Prognostics Center of Excellence data repository and unzip it — you'll get
   `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt` for X in 1–4.
2. Put those files in `data/raw/CMAPSS/` (create the folder if needed).
3. Run the loader instead of the synthetic generator:
   ```bash
   python scripts/load_cmapss.py --raw-dir data/raw/CMAPSS --dataset FD001
   python -m src.ml.train
   streamlit run dashboard/app.py
   ```
   This writes the same `data/mock/*.csv` files (training set from the full
   run-to-failure trajectories, fleet snapshot from the partial test
   trajectories), plus `data/mock/test_true_rul.csv` — the ground-truth RUL
   for each test unit, for checking prediction accuracy. `MACHINE_NAME` /
   `LINE` / maintenance / production history are still synthesized, since
   C-MAPSS has no equivalent of those.

Click a machine's agentic action buttons (Create Work Order / Schedule Repair
/ Recommend Parts) to see them logged to `data/mock/action_outcomes.csv` and
appear in the "Outcome log" section — that's the feedback-loop stub.

### Running the tests

```bash
python -m pytest tests/ -v
```
22 tests, no Snowflake/model-file dependency — pure-logic checks on OEE math,
the feature-engineering slope formula, and the RUL/risk/confidence helpers.

## Switching to real Snowflake

**Status: verified working, but against the wrong account.** The pipeline
(tables, real C-MAPSS data, SQL feature engineering, ML predictions) is
confirmed running live against Snowflake — but the account used to verify
this (`ci12831...`) was created via the generic `signup.snowflake.com` trial,
not the official hackathon sign-up link from the Contest Site. Per the
hackathon's own docs, that matters: Cortex/CoCo CLI access is unreliable on
a self-made trial. **Before submission, re-point `.env` at an account
created through the actual Hack2Skill contest flow**, then re-run steps 3–4
below against it (fast — the scripts are already written).

1. Get the Snowflake sign-up link from the Hack2Skill Contest Site (after
   registering for the hackathon) — not the generic trial signup.
2. Fill in `.env`: `SNOWFLAKE_ACCOUNT` (format `<account_locator>.<region>.<cloud>`,
   e.g. `ci12831.ap-southeast-7.aws` — found in your Snowsight URL), `SNOWFLAKE_USER`,
   `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE`, then set
   `SNOWFLAKE_MODE=snowflake`.
3. Run `sql/001_create_tables.sql` against your account (Snowsight worksheet,
   or `snowsql -f sql/001_create_tables.sql`).
4. Load data into the tables — e.g. via Snowpark
   (`session.create_dataframe(df).write.mode("overwrite").save_as_table(...)`)
   from the `data/mock/*.csv` files `load_cmapss.py` already produced.
5. Everything else — feature engineering, training, the dashboard — runs
   unchanged; `src/decision_layer.py` calls `SNOWFLAKE.CORTEX.COMPLETE`
   instead of the mock reasoning template.

### Two environment issues you may hit (both already handled in code)

**TLS certificate verification fails on a network with SSL-inspecting
firewalls/proxies.** The Snowflake connector uses its own bundled OpenSSL
(via pyOpenSSL) for certificate checks — it never consults the OS trust
store, so it rejects a proxy's substituted certificate even when macOS
already trusts it. `src/connection.py`'s `_ensure_local_network_ca_trusted()`
pulls whatever's already trusted in the macOS Keychain into a combined CA
bundle for the connector to use (`.local_ca_bundle.pem`, gitignored,
regenerated automatically) — it does **not** disable verification, it just
gives the connector the same trust the OS already has. No-op on any machine
without such a proxy.

**`SNOWFLAKE.CORTEX.COMPLETE` isn't available on trial-tier accounts.**
`explain()` in `src/decision_layer.py` catches this and falls back to the
templated reasoning (prefixed with `[Cortex unavailable: ...]` so it's
obvious in the UI) instead of crashing the dashboard. Upgrading the account
tier is enough to get real Cortex reasoning with no code changes.

### Cortex Analyst (natural-language Q&A) — built, not yet verified live

`src/cortex_analyst.py` + `sql/003_semantic_model.yaml` add an "Ask the fleet
a question" panel (Snowflake mode only) using Cortex Analyst's REST API over
a Semantic View. This was built from Snowflake's official docs but **has
never been run against a live account** — no account with Cortex Analyst
access has existed yet while building it. Three things to do once the right
account exists:
1. Create the Semantic View — see `sql/004_create_semantic_view.sql`
   (Snowsight UI import is the reliable path; the SQL stored-procedure call
   in that file is unverified).
2. Confirm the role has `SNOWFLAKE.CORTEX_USER` or `SNOWFLAKE.CORTEX_ANALYST_USER`.
3. Confirm the account's region supports Cortex Analyst natively, or has
   cross-region inference enabled (`ALTER ACCOUNT SET
   CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';` — ACCOUNTADMIN only). Native
   regions: AWS ap-northeast-1/ap-southeast-2/us-east-1/us-west-2/eu-central-1/
   eu-west-1, Azure East US 2/West Europe.

If any of this is wrong or incomplete, the panel fails with a visible
"Cortex Analyst unavailable: ..." message — same fail-safe pattern as
`decision_layer.py` — rather than crashing the dashboard.

## Layout

| Path | Diagram component |
|---|---|
| `sql/001_create_tables.sql` | Snowflake tables |
| `sql/002_analysis_worksheet.sql` | Standalone business-question Worksheet — paste into Snowsight |
| `sql/003_semantic_model.yaml`, `sql/004_create_semantic_view.sql` | Cortex Analyst Semantic View definition |
| `scripts/generate_mock_data.py` | Stand-in for the data sources, until Snowflake is live |
| `src/feature_engineering.py` | Feature Engineer (sensor trends, rolling stats, degradation) |
| `src/oee.py` | OEE Calculation (availability, performance, quality) |
| `src/ml/train.py`, `src/ml/predict.py` | ML Prediction (RUL, failure probability, risk, confidence) |
| `src/decision_layer.py` | AI / Decision Layer (Cortex COMPLETE in Snowflake mode) |
| `src/cortex_analyst.py` | Natural-language Q&A over the business tables (Cortex Analyst) |
| `src/business_impact.py` | Predictions → estimated $ downtime cost avoided |
| `dashboard/app.py` | Unified Command Center + agentic actions |
| `src/outcomes.py`, `src/data_access.py` | Outcome logging → `ACTION_OUTCOMES` feedback loop |
| `tests/` | pytest unit tests (OEE, feature engineering, ML helpers, business impact) |

## Known simplifications (by design, not overbuilt)

- The feedback loop logs outcomes but does not auto-retrain — closing that
  loop into continuous learning is future work, flagged as a foundation.
- `src/ml/train.py` trains on whatever's in `training_run_to_failure.csv` —
  real NASA run-to-failure trajectories by default (via `load_cmapss.py`),
  or synthetic ones if you ran `generate_mock_data.py` instead. Point it at
  real historical `RAW_SENSOR_DATA` + failure events once that history
  exists for an actual fleet in Snowflake.
- Agentic actions ("Create Work Order" etc.) log to `ACTION_OUTCOMES` only —
  intentionally not wired to a real CMMS/ticketing system, since none exists
  for this demo to integrate with.

## Hackathon submission checklist

Built for the [Snowflake CoCo CLI Hackathon 2026 — GCC Edition](https://hack2skill.com/event/cococlihack-gccedition/)
([Terms & Conditions](https://docs.google.com/document/d/e/2PACX-1vTrXSK6v7T9tP3-Ab8LuFCDOuuW90debariK5I3PsIF0TrQ4A6q5RSC2B2wA4WM7Qif16AgynvdA4XL/pub)).
Judging criteria per T&C §9: theme-responsive prototype including use of
Cortex Code CLI, Python/Java/Scala, required Snowflake platform use, special
consideration for Snowpark/Worksheets/Streamlit/Marketplace.

- [x] Problem statement 3 (Predictive Maintenance and OEE Command Center)
- [x] Working end-to-end prototype (not a notebook) — Streamlit dashboard
- [x] Snowflake platform genuinely used — tables, Snowpark, SQL push-down
- [x] Snowpark ✓ and Streamlit ✓ (both named for "special consideration")
- [x] Python
- [x] Real dataset (NASA C-MAPSS) with sources documented — [DATA_SOURCES.md](DATA_SOURCES.md)
- [x] Snowflake Worksheet — [sql/002_analysis_worksheet.sql](sql/002_analysis_worksheet.sql)
- [x] GitHub repository — https://github.com/Anushree-DK/predictive-maintenance-oee
- [x] Unit tests (22, pytest) — signals engineering rigor beyond the minimum
- [x] Business $-impact framing, not just ML metrics (`src/business_impact.py`)
- [~] **CoCo CLI** — installed, connection-tested, confirmed blocked
      specifically by account type (not tooling); not yet actually used to
      build anything, since that needs the account below first
- [~] **Cortex Analyst NL Q&A** — built (`src/cortex_analyst.py` + Semantic
      View YAML), not yet live-tested — same account blocker
- [ ] **Correct Snowflake account** — currently on a self-made trial, not the
      official Hack2Skill contest sign-up; Cortex/CoCo need the latter
- [ ] **Snowflake Marketplace** — deliberately not started: acquiring a
      listing on the wrong (soon-to-be-replaced) account wastes the effort;
      revisit once the correct account is live
- [ ] Presentation deck (PPT or similar) — required for submission
- [ ] Team/participant profile submitted on Hack2Skill
- [ ] Demo screenshot/GIF in this README

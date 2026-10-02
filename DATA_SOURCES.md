# Data Sources

**Every table in this project comes from one real dataset: NASA's C-MAPSS.** Nothing
is randomly generated. Each column is either a NASA reading or a deterministic
derivation of one. The only numbers that are not measured are the cost and duration
rates listed at the end, and they are labelled as assumptions everywhere they appear.

## NASA Turbofan Engine Degradation Simulation Data Set (C-MAPSS)

- **Source**: NASA Prognostics Center of Excellence (PCoE) Data Set Repository —
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  (direct archive used by `README.md`: `phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`)
- **Citation**: A. Saxena, K. Goebel, D. Simon, and N. Eklund (2008). "Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proc. 1st
  International Conference on Prognostics and Health Management (PHM08).
- **What it is**: 21 sensors and 3 operational settings, recorded once per flight
  cycle, for 1,416 turbofan engines in four subsets. NASA produced it with its
  high-fidelity C-MAPSS engine simulator. It is not field data from an airline
  (no public dataset is), but it is the standard benchmark for remaining-useful-life
  prediction and is what published results are compared against.
- **License/terms**: no formal license is stated. The repository carries a liability
  disclaimer and asks users to acknowledge the repository and the data donors. No
  commercial-use or redistribution restriction is stated.

| Subset | Engines (failed / in service) | Operating conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 / 100 | 1 (sea level) | HPC degradation |
| FD002 | 260 / 259 | 6 | HPC degradation |
| FD003 | 100 / 100 | 1 (sea level) | HPC + fan degradation |
| FD004 | 249 / 248 | 6 | HPC + fan degradation |

## How each table is derived (`src/cmapss.py`)

| Table | Derivation |
|---|---|
| `MACHINE_DATA` | One row per engine. `FLEET` is the subset. `STATUS = FAILED` for *train* trajectories (they run until failure) and `IN_SERVICE` for *test* trajectories (they stop before failure). Conditions and fault modes come from NASA's readme. |
| `RAW_SENSOR_DATA` | NASA's readings, unchanged. Only the engine ID column is added. |
| `MAINTENANCE_HISTORY` | One `UNPLANNED_FAILURE` event per failed engine, at the last cycle of its trajectory. That is the cycle at which NASA's simulation reached failure. |
| `OPERATING_PERIODS` | Per engine, per 10-cycle period: cycles flown, cycles in spec (every informative sensor within 3σ of the healthy baseline, i.e. failed engines' first 30 cycles, per operating regime), the average health index (a linear model over the sensors, fitted on failed engines to run from 1 when healthy to 0 at failure), and whether the engine failed in that period. |
| `REGIME_SENSOR_STATS` | Per-regime sensor mean and std, computed from failed engines only. The regime is `ROUND(OP_SETTING_1)`, the flight altitude in kft, which separates C-MAPSS's six operating conditions exactly. |
| `FLEET_GROUND_TRUTH` | NASA's `RUL_FD00X.txt`: the true remaining life of each in-service engine. **Used only to evaluate the model**. The dashboard and model never read it as an input. |
| `ACTION_OUTCOMES` | Starts empty. Rows come only from people using the dashboard's action buttons. |

Everything the model learns from comes from failed engines. Everything it is
evaluated on comes from in-service engines that it never saw during training.

## OEE, adapted to an engine fleet (`src/oee.py`)

C-MAPSS has no production counts, so OEE's three factors are mapped to what an
engine fleet actually measures:

- **Availability** = flight hours / (flight hours + unplanned repair downtime). A
  period containing a real failure carries `AVG_UNPLANNED_REPAIR_HOURS` of downtime.
- **Performance** = the period's average health index (the share of healthy-engine
  performance retained).
- **Quality** = the share of cycles flown with every informative sensor in spec.

## Assumptions (the only non-measured numbers)

All of these live in `config.py`.

- **`COST_PER_DOWNTIME_HOUR_USD` ($25,000)**: MaintainX's 2024 State of Industrial
  Maintenance report, the survey average cost of one hour of unplanned downtime.
  Published estimates range from about $25K to $260K+ per hour; this uses the
  conservative end.
- **`AVG_UNPLANNED_REPAIR_HOURS` (18) / `AVG_PLANNED_REPAIR_HOURS` (4)**: not taken
  from a study. Editable parameters for the gap between an emergency repair and a
  scheduled maintenance window.
- **`HOURS_PER_CYCLE` (2.0)**: one C-MAPSS cycle is one flight; 2.0 is an assumed
  average flight duration. Only availability uses it.

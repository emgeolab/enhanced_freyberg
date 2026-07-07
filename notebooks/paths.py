
from pathlib import Path
"""

Layout
------
PROJECT_ROOT/
├─ notebooks/   ← this file + 01–05.ipynb + all *.py helpers
├─ reference/   ← read-only input model (01 extracts the ZIP here; 02 reads it)
├─ runs/        ← heavy, regenerable MF6 / PEST workspaces
│  ├─ truth/          base_truth_template · truth_candidates · final_truth   (01)
│  ├─ pest_template/  built by 02        → read by 03·04·05
│  ├─ pest_master/    GLM results by 03  → read by 04·05
│  ├─ sweep_master/   sweep results      (05)
│  └─ _workers/       parallel-run scratch; safe to delete
└─ results/     ← small, paper-facing deliverables
   ├─ truth/          01 outputs: synthetic obs/forecasts + npz + metadata
   └─ tables/         03·04·05 result tables
"""

# --- project root -----------------------------------------------------------
NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
# To pin a different root instead, uncomment:
# PROJECT_ROOT = Path(r"C:\workspace\gwmodeling\model_Freyberg_3L_tr")

# --- top-level buckets -------------------------------------------------------
REFERENCE_DIR = PROJECT_ROOT / "reference"
RUNS_DIR      = PROJECT_ROOT / "runs"
RESULTS_DIR   = PROJECT_ROOT / "results"

# --- runs/ : regenerable workspaces -----------------------------------------
TRUTH_RUN_DIR  = RUNS_DIR / "truth"                    # 01
BASE_TRUTH_WS  = TRUTH_RUN_DIR / "base_truth_template"
CANDIDATE_DIR  = TRUTH_RUN_DIR / "truth_candidates"
FINAL_TRUTH_WS = TRUTH_RUN_DIR / "final_truth"

PEST_TEMPLATE  = RUNS_DIR / "pest_template"            # 02 → 03·04·05
PEST_MASTER    = RUNS_DIR / "pest_master"              # 03 → 04·05
SWEEP_MASTER   = RUNS_DIR / "sweep_master"             # 05

WORKERS_DIR       = RUNS_DIR / "_workers"              # temp; gitignore
GLM_WORKER_ROOT   = WORKERS_DIR / "glm"                # 03
SWEEP_WORKER_ROOT = WORKERS_DIR / "sweep"              # 05

# --- results/ : small deliverables ------------------------------------------
TRUTH_RESULTS_DIR = RESULTS_DIR / "truth"              # 01 outputs
TABLES_DIR        = RESULTS_DIR / "tables"             # 03·04·05

# --- reference/ : extracted input model -------------------------------------
REFERENCE_ZIP      = REFERENCE_DIR / "mf6_freyberg.zip"
REFERENCE_EXTRACT  = REFERENCE_DIR / "mf6_freyberg_reference"
REFERENCE_TEMPLATE = REFERENCE_EXTRACT / "template"

# --- cross-notebook handoff files (single source of truth) ------------------
PARAM_ENSEMBLE_NPZ = TRUTH_RESULTS_DIR / "truth_parameter_ensembles.npz"
CANDIDATE_SUMMARY  = TRUTH_RESULTS_DIR / "truth_candidate_summary.csv"
HISTORY_OBS_CSV    = TRUTH_RESULTS_DIR / "synthetic_history_observations.csv"  # → 02
FORECASTS_CSV      = TRUTH_RESULTS_DIR / "synthetic_forecasts.csv"             # → 04·05
TRUTH_METADATA     = TRUTH_RESULTS_DIR / "truth_metadata.json"

# --- create the standing buckets on import ----------------------------------
def ensure_dirs():
    """Create the persistent folders (not the per-run workspaces, which each
    notebook resets on its own)."""
    for d in (REFERENCE_DIR, RUNS_DIR, RESULTS_DIR,
              TRUTH_RUN_DIR, TRUTH_RESULTS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

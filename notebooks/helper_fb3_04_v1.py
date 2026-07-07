from __future__ import annotations
import numpy as np
import pandas as pd


def forecast_table(la):
    """Prior/posterior forecast standard deviations and percent uncertainty reduction (Schur)."""
    s = la.get_forecast_summary()
    out = pd.DataFrame({"prior_std": np.sqrt(s["prior_var"]),
                        "post_std": np.sqrt(s["post_var"])})
    out["percent_reduction"] = 100.0 * (1.0 - out["post_std"] / out["prior_std"])
    return out


def to_summary_df(summary, means):
    """Reshape a pyEMU variance summary into the columns pyemu.plot_utils expects."""
    means = pd.Series(means).reindex(summary.index)
    return pd.DataFrame({
        "prior_mean": means.values,
        "prior_stdev": np.sqrt(summary["prior_var"].values),
        "post_mean": means.values,
        "post_stdev": np.sqrt(summary["post_var"].values),
    }, index=summary.index)


def param_reduction_by_group(la):
    """Mean prior/posterior parameter std and percent reduction, by parameter group."""
    ps = la.get_parameter_summary()
    ps["prior_std"] = np.sqrt(ps["prior_var"])
    ps["post_std"] = np.sqrt(ps["post_var"])
    ps["group"] = [p.split("_pp")[0] if "_pp" in p else p.rsplit("_", 1)[0] for p in ps.index]
    g = ps.groupby("group")[["prior_std", "post_std"]].mean()
    g["percent_reduction"] = 100.0 * (1.0 - g["post_std"] / g["prior_std"])
    return g


def load_truth_forecasts(path, forecast_names):
    """Read the truth forecast values (validation only) and align to the pst forecast names.

    Looks up tailwater / headwater / gw_3 rows in synthetic_forecasts.csv and maps them onto the
    pst forecast names (tailwater_sp13, headwater_sp22, gw_3_sp22). Returns a Series.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    val_col = next((c for c in ["value", "truth", "observed", "forecast_value", "modelled"]
                    if c in df.columns), df.columns[-1])
    key_col = next((c for c in ["forecast", "name", "obsnme", "key"] if c in df.columns), df.columns[0])
    truth = {}
    for nm in forecast_names:
        stem = nm.split("_sp")[0]                       # tailwater / headwater / gw_3
        hit = df[df[key_col].astype(str).str.lower().str.contains(stem)]
        truth[nm] = float(hit[val_col].iloc[0]) if len(hit) else np.nan
    return pd.Series(truth)

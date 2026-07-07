

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- Table 1 -> base + bounds
def property_table():
    """Prior-mean base value (geomean of bounds) and multiplier bounds per property/layer.

    Because base = sqrt(lb*ub), every multiplier range is [sqrt(lb/ub), sqrt(ub/lb)] = [0.1, 10]
    here (all properties span two log10 orders), so log10 sigma = (1-(-1))/4 = 0.5.
    """
    specs = [
        ("hk_l1", 0, 0.3, 30.0), ("hk_l2", 1, 0.03, 3.0), ("hk_l3", 2, 3.0, 300.0),
        ("vk_l1", 0, 0.03, 3.0), ("vk_l2", 1, 0.003, 0.3), ("vk_l3", 2, 0.3, 30.0),
        ("ss_l1", 0, 1e-7, 1e-5), ("ss_l2", 1, 1e-7, 1e-5), ("ss_l3", 2, 1e-7, 1e-5),
        ("sy_l1", 0, 1e-3, 0.1),
    ]
    df = pd.DataFrame(specs, columns=["name", "layer", "lower", "upper"])
    df["base"] = np.sqrt(df["lower"] * df["upper"])                      # prior-mean homogeneous
    df["mult_lb"] = df["lower"] / df["base"]                             # multiplier lower bound
    df["mult_ub"] = df["upper"] / df["base"]
    df["log10_std"] = (np.log10(df["upper"]) - np.log10(df["lower"])) / 4.0
    return df


def time_mult_bounds():
    """Multiplier bounds (around 1) for the recharge and per-well pumping time series."""
    rch = dict(base=6.169e-5, lower=2.938e-5, upper=8.813e-5)
    wel = dict(base=157.5, lower=43.75, upper=306.25)
    out = {}
    for key, d in [("rch", rch), ("wel", wel)]:
        lb, ub = d["lower"] / d["base"], d["upper"] / d["base"]
        out[key] = dict(mult_lb=lb, mult_ub=ub,
                        log10_std=(np.log10(ub) - np.log10(lb)) / 4.0)
    return out


# --------------------------------------------------------------------------- observation ordering
def obs_order(history_sp):
    """Fixed observation name order produced by forward_run.py.

    History (weight>0): for each SP in history_sp -> sw_1, gw_1, gw_2.
    Forecasts (weight=0): tailwater_sp13, headwater_sp22, gw_3_sp22.
    """
    names = []
    for sp in history_sp:
        names += [f"sw_1_sp{sp:02d}", f"gw_1_sp{sp:02d}", f"gw_2_sp{sp:02d}"]
    forecasts = ["tailwater_sp13", "headwater_sp22", "gw_3_sp22"]
    return names, forecasts


def read_history_csv(path):
    """Read synthetic_history_observations.csv -> {obsnme: (observed, weight)} (truth not touched)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return {str(r["obsnme"]).lower(): (float(r["observed"]), float(r["weight"]))
            for _, r in df.iterrows()}


# --------------------------------------------------------------------------- prior covariance
def spatial_cov_from_pp(pp_df, geostruct):
    """pyEMU spatial prior covariance for one pilot-point set from a GeoStruct."""
    import pyemu
    return geostruct.covariance_matrix(pp_df.x, pp_df.y, pp_df.parnme)


def temporal_cov(names, n, range_months, log10_std):
    """Exponential temporal prior covariance (Cov) for a multiplier time series."""
    import pyemu
    idx = np.arange(n, dtype=float)
    C = np.exp(-np.abs(idx[:, None] - idx[None, :]) / float(range_months)) * (log10_std ** 2)
    return pyemu.Cov(x=C, names=list(names))


# --------------------------------------------------------------------------- plots
def plot_pilot_points(sr, df_pp, idomain_layer, prefix, title=""):
    sub = df_pp[df_pp.parnme.str.startswith(prefix)]
    fig, ax = plt.subplots(figsize=(4.5, 7))
    ax.imshow(np.where(idomain_layer > 0, 1.0, np.nan), cmap="Greys", alpha=0.25)
    ax.scatter(sub.j, sub.i, s=25, c="red", edgecolor="k")
    ax.set_title(title or f"{prefix} pilot points  (n={len(sub)})")
    ax.set_xlabel("column"); ax.set_ylabel("row"); plt.tight_layout()
    return fig


def plot_variogram(range_m, max_h=4000):
    h = np.linspace(0, max_h, 200)
    gamma = 1.0 - np.exp(-h / range_m)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h, gamma); ax.axvline(range_m, color="r", ls="--", label=f"range {range_m:.0f} m")
    ax.set_xlabel("separation (m)"); ax.set_ylabel("semivariance (unit sill)")
    ax.set_title("exponential variogram"); ax.legend(); plt.tight_layout()
    return fig


def plot_cov(cov, title=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cov.as_2d, cmap="viridis"); plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title); plt.tight_layout()
    return fig

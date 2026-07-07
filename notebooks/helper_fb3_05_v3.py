from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================== ensemble machinery
def draw_posterior_ensemble(pst, post_cov, num_reals=200, seed=0):
    """Draw `num_reals` from N(calibrated parval1, posterior_cov), enforce bounds. Returns ensemble."""
    import pyemu
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=post_cov, num_reals=num_reals)
    pe.enforce()
    return pe


def filter_lowest_phi(sweep_out, n_keep=50, phi_col=None):
    """Return the filtered sweep-out rows with the lowest objective function (failed runs dropped)."""
    df = sweep_out.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "failed_flag" in df.columns:
        df = df[df["failed_flag"] == 0]
    if phi_col is None:
        phi_col = "meas_phi" if "meas_phi" in df.columns else ("phi" if "phi" in df.columns else None)
    df = df.sort_values(phi_col).head(n_keep)
    return df, phi_col


def reconstruct_fields(pst, par_series, master_dir, props, shape, rch_base, wel_base, nper,
                       fac_file="pp.fac"):
    """Rebuild one realization's property fields from its parameter values (dict: hk/vk/ss/sy)."""
    import os, pyemu
    pst.parameter_data.loc[par_series.index, "parval1"] = par_series.values.astype(float)
    pst.write_input_files(pst_path=master_dir)
    fields = {}
    for p in props:
        mult = np.asarray(pyemu.geostats.fac2real(os.path.join(master_dir, f"{p}pp.dat"),
                                                  factors_file=os.path.join(master_dir, fac_file),
                                                  out_file=None)).reshape(shape)
        tag, lay = p.split("_")[0], p.split("_")[1]
        base = np.loadtxt(os.path.join(master_dir, f"{tag}_base_{lay}.dat"))
        fields[p] = base * mult
    return {"hk": np.stack([fields[f"hk_l{k+1}"] for k in range(3)]),
            "vk": np.stack([fields[f"vk_l{k+1}"] for k in range(3)]),
            "ss": np.stack([fields[f"ss_l{k+1}"] for k in range(3)]),
            "sy": fields["sy_l1"]}


# =============================================================== obs vs sim (ensemble time series)
def plot_ensemble_timeseries(sweep_filt, pst, history_sp):
    """obs vs sim, 3x1. Ensemble realizations = red lines (no markers); observed = blue + markers.

    sweep_filt : filtered ensemble (columns lower-cased; obs values per realization).
    pst        : control file (observed obsval).
    history_sp : list of history stress periods (e.g. range(2, 14)).
    """
    obs = pst.observation_data
    series = [("sw_1", "m$^3$/d"), ("gw_1", "m"), ("gw_2", "m")]
    label_fs = 18
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    for ax, (pre, unit) in zip(axes, series):
        names = [f"{pre}_sp{sp:02d}" for sp in history_sp]
        # ensemble (simulated) -- red lines, NO markers
        first = True
        for _, row in sweep_filt.iterrows():
            y = [row.get(n.lower(), np.nan) for n in names]
            ax.plot(history_sp, y, "-", color="red", alpha=0.30, lw=0.6,
                    label="Simulated" if first else "_nolegend_")
            first = False
        # observed -- blue with markers
        obsv = [float(obs.loc[obs.obsnme == n, "obsval"].iloc[0]) if (obs.obsnme == n).any()
                else np.nan for n in names]
        ax.plot(history_sp, obsv, "-", marker="o", color="black", lw=1.9, ms=4, label="Observed")
        ax.set_title(f"{pre} obs vs sim", fontsize=label_fs)
        ax.set_ylabel(f"{pre} ({unit})", fontsize=label_fs)
        ax.set_xlim(0, max(history_sp))
        ax.set_xticks(range(0, max(history_sp) + 1, 1))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, axis="both", alpha=0.3)
    axes[-1].set_xlabel("Stress period", fontsize=label_fs)
    axes[0].legend()
    return fig


# =============================================================== objective function (standalone)
def plot_objective_hist(sweep_filt, phi_col="meas_phi"):
    """Objective-function histogram of the filtered ensemble, standalone, square aspect."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(sweep_filt[phi_col].astype(float), bins=15, color="tab:blue", edgecolor="k")
    ax.set_title("objective function")
    ax.set_xlabel("phi"); ax.set_ylabel("count")
    ax.set_box_aspect(1)
    plt.tight_layout()
    return fig


# =============================================================== forecast prior/posterior distributions
def plot_forecast_distributions(fs, truth_fore, fore_names=None, sweep_filt=None, forecast_meta=None):
    """1x3 FOSM prior (gray) / posterior (blue) gaussians + truth (red); optional MC histogram bars.

    fs         : forecast summary with 'prior_std','post_std','post_mean' (index = forecast names).
    truth_fore : Series of truth forecast values.
    sweep_filt : if given, overlay the Monte Carlo histogram (ensemble forecast values) as bars.
    """
    if fore_names is None:
        fore_names = list(fs.index)
    if forecast_meta is None:
        forecast_meta = {
            "tailwater_sp13": ("Tailwater forecast (sp13)", r"SW-GW flux (m$^3$/d)"),
            "headwater_sp22": ("Headwater forecast (sp22)", r"SW-GW flux (m$^3$/d)"),
            "gw_3_sp22":      ("gw_3 forecast (sp22)",      "Groundwater level (m)"),
        }
    prior_color, posterior_color, truth_color = "0.60", "#94ACEE", "#E53945"

    def gaussian_pdf(x, mean, stdev):
        return np.exp(-0.5 * ((x - mean) / stdev) ** 2) / (stdev * np.sqrt(2.0 * np.pi))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), gridspec_kw={"wspace": 0.38})
    for i, (ax, nm) in enumerate(zip(axes, fore_names)):
        post_mean = float(fs.loc[nm, "post_mean"])
        prior_mean = post_mean
        prior_std = float(fs.loc[nm, "prior_std"])
        post_std = float(fs.loc[nm, "post_std"])
        truth = float(truth_fore[nm])

        lo = min(prior_mean - 3 * prior_std, post_mean - 3 * post_std, truth)
        hi = max(prior_mean + 3 * prior_std, post_mean + 3 * post_std, truth)
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        x = np.linspace(lo - pad, hi + pad, 500)

        if sweep_filt is not None and nm.lower() in sweep_filt.columns:
            ax.hist(sweep_filt[nm.lower()].astype(float), bins=15, density=True,
                    color=posterior_color, alpha=0.30, edgecolor="k", zorder=0,
                    label="Monte Carlo" if i == 0 else "_nolegend_")

        ax.fill_between(x, 0, gaussian_pdf(x, prior_mean, prior_std), color=prior_color,
                        alpha=0.38, label="Prior" if i == 0 else "_nolegend_", zorder=1)
        ax.fill_between(x, 0, gaussian_pdf(x, post_mean, post_std), color=posterior_color,
                        alpha=0.55, label="Posterior" if i == 0 else "_nolegend_", zorder=2)
        ax.axvline(truth, color=truth_color, lw=2.8,
                   label="Truth" if i == 0 else "_nolegend_", zorder=3)

        title, xlabel = forecast_meta.get(nm, (nm, ""))
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
        ax.set_ylabel("Probability density", fontsize=12, labelpad=8)
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=10, top=True, direction="in", length=5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4,
               frameon=False, fontsize=12, handlelength=2.2, columnspacing=2.2)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.82)
    return fig


# =============================================================== one realization's fields (4x3 jet)
def plot_realization_fields(fields, idomain, vmins=None, vmaxs=None):
    """One realization's 4x3 property fields on the user's exact template (cmap='jet').

    fields  : reconstruct_fields output (dict with 'hk','vk','ss' (3,r,c) and 'sy' (r,c)).
    idomain : (3, nrow, ncol) integer idomain.
    vmins/vmaxs : optional fixed log ranges; default = these fields' 2/98 percentiles.
    """
    final_hk, final_vk, final_ss, final_sy = fields["hk"], fields["vk"], fields["ss"], fields["sy"]

    fig, axes = plt.subplots(4, 3, figsize=(8, 12), constrained_layout=True)
    fields = [final_hk, final_vk, final_ss, final_sy]
    param_names = [r"$\log_{10} K$", r"$\log_{10} K_{33}$", r"$\log_{10} S_s$", r"$\log_{10} S_y$"]

    if vmins is None or vmaxs is None:
        vmins, vmaxs = [], []
        for i in range(4):
            if i == 3:
                arr = np.log10(fields[i]); arr[idomain[0, :, :] <= 0] = np.nan
            else:
                arr = np.log10(fields[i]); arr[idomain <= 0] = np.nan
            vals = arr[np.isfinite(arr)]
            vmins.append(np.percentile(vals, 2)); vmaxs.append(np.percentile(vals, 98))

    for i in range(4):
        for j in range(3):
            ax = axes[i, j]
            if i == 3:
                if j != 0:
                    ax.axis("off"); continue
                field = np.log10(fields[i]).copy(); field[idomain[0, :, :] <= 0] = np.nan
            else:
                field = np.log10(fields[i][j]).copy(); field[idomain[j, :, :] <= 0] = np.nan
            im = ax.imshow(field, origin="upper", vmin=vmins[i], vmax=vmaxs[i], cmap="jet")
            ax.tick_params(axis="both", which="major", labelsize=10)
            if i == 3:
                ax.set_title(param_names[i])
            else:
                ax.set_title(f"{param_names[i]}, L{j+1}")
            ax.set_xlabel("column", fontsize="11"); ax.set_ylabel("row", fontsize="11")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad=0.05)
            fig.colorbar(im, cax=cax)
    axes[3, 1].axis("off"); axes[3, 2].axis("off")
    return fig

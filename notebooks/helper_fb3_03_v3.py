from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_phi(master_dir, case):
    """Measurement phi vs GLM iteration (not total phi, which swings with the reg weight)."""
    iobj = pd.read_csv(os.path.join(master_dir, f"{case}.iobj"))
    col = "measurement_phi" if "measurement_phi" in iobj.columns else \
          ("measurement" if "measurement" in iobj.columns else iobj.columns[-1])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(iobj["iteration"], iobj[col], "-o")
    ax.set_xlabel("GLM iteration"); ax.set_ylabel("measurement phi")
    ax.set_title("objective function history"); ax.grid(alpha=0.3)
    plt.tight_layout()
    return iobj


def plot_time_series_fit(pst, n_hist=None):
    """Observed vs simulated per series (sw_1, gw_1, gw_2) over the history window only."""
    
    res = pst.res.set_index("name")
    series = [("sw_1", "m$^3$/d"), ("gw_1", "m"), ("gw_2", "m")]
    label_fs = 18

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)

    for ax, (pre, unit) in zip(axes, series):
        names = sorted(
            [n for n in res.index if n.startswith(pre)],
            key=lambda n: int(n.split("sp")[-1])
        )

        sp = [int(n.split("sp")[-1]) for n in names]

        # observed / simulated: 선 + 실제 데이터 지점 marker
        ax.plot(
            sp, res.loc[names, "measured"],
            "-", marker="o", color="black", lw=1.7, ms=4,
            label="Observed"
        )
        ax.plot(
            sp, res.loc[names, "modelled"],
            "-", marker="o", color="red", lw=1.7, ms=4,
            label="Simulated"
        )

        ax.set_title(f"{pre} obs vs sim", fontsize=label_fs)
        ax.set_ylabel(f"{pre} ({unit})", fontsize=label_fs)

        # x축: 0부터 시작, stress period 1 단위 모두 표시
        ax.set_xlim(0, max(sp))
        ax.set_xticks(range(0, max(sp) + 1, 1))

        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, axis="both", alpha=0.3)

    axes[-1].set_xlabel("Stress period", fontsize=label_fs)

    # 필요 시 범례는 첫 번째 패널에만 표시
    axes[0].legend()

    return fig


def plot_one_to_one(pst):
    """1:1 observed vs simulated for all weighted observations (large, sharp, red)."""
    res = pst.res.set_index("name")
    w = res[res["weight"] > 0]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(w["measured"], w["modelled"], s=70, c="red", edgecolor="k",
               linewidth=0.7, alpha=0.9, zorder=3)
    lo = min(w["measured"].min(), w["modelled"].min())
    hi = max(w["measured"].max(), w["modelled"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, zorder=2)
    ax.set_xlabel("observed", fontsize=15)
    ax.set_ylabel("simulated", fontsize=15)
    ax.set_title("1:1 fit (weighted obs)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()
    return fig


def regenerate_fields_3l(master_dir, props, shape, fac_file="pp.fac"):
    """Calibrated property field per prop = base x kriged best-multiplier."""
    import pyemu
    fields = {}
    for p in props:
        mult = pyemu.geostats.fac2real(os.path.join(master_dir, f"{p}pp.dat"),
                                       factors_file=os.path.join(master_dir, fac_file),
                                       out_file=None)
        mult = np.asarray(mult).reshape(shape)
        tag, lay = p.split("_")[0], p.split("_")[1]
        base = np.loadtxt(os.path.join(master_dir, f"{tag}_base_{lay}.dat"))
        fields[p] = base * mult
    return fields


def field_log_ranges(fields, idomain):
    """Notebook-1's per-parameter 2/98 log10 colour ranges, in [HK, VK, SS, SY] order.

    Call this on the TRUTH fields and pass the returned (vmins, vmaxs) to plot_calibrated_fields
    so the calibrated figure uses the exact same colour scale as notebook 1.
    """
    stacks = [np.stack([fields[f"hk_l{k+1}"] for k in range(3)]),
              np.stack([fields[f"vk_l{k+1}"] for k in range(3)]),
              np.stack([fields[f"ss_l{k+1}"] for k in range(3)]),
              fields["sy_l1"]]
    vmins, vmaxs = [], []
    for i in range(4):
        arr = np.log10(stacks[i]).copy()
        if i == 3:
            arr[idomain[0, :, :] <= 0] = np.nan
        else:
            arr[idomain <= 0] = np.nan
        vals = arr[np.isfinite(arr)]
        vmins.append(np.percentile(vals, 2))
        vmaxs.append(np.percentile(vals, 98))
    return vmins, vmaxs


def plot_calibrated_fields(fields, idomain, vmins=None, vmaxs=None):
    """Notebook-1 4x3 template (HK/VK/SS by layer + SY L1) drawn with the calibrated values.

    Titles / labels / positions are exactly as in notebook 1. Pass vmins/vmaxs (e.g. from
    field_log_ranges(truth_fields, idomain)) to use notebook-1's colour scale; if omitted, the
    ranges are taken from these fields' 2/98 percentiles (which collapse when near-homogeneous).
    """
    final_hk = np.stack([fields[f"hk_l{k+1}"] for k in range(3)])
    final_vk = np.stack([fields[f"vk_l{k+1}"] for k in range(3)])
    final_ss = np.stack([fields[f"ss_l{k+1}"] for k in range(3)])
    final_sy = fields["sy_l1"]

    fig, axes = plt.subplots(4, 3, figsize=(8, 12), constrained_layout=True)
    fields = [
        final_hk,
        final_vk,
        final_ss,
        final_sy
    ]
    param_names = [
        r"$\log_{10} K$",
        r"$\log_{10} K_{33}$",
        r"$\log_{10} S_s$",
        r"$\log_{10} S_y$"
    ]
    if vmins is None or vmaxs is None:
        vmins = []
        vmaxs = []
        for i in range(4):
            if i == 3:
                arr = np.log10(fields[i])
                arr[idomain[0, :, :] <= 0] = np.nan
                vals = arr[np.isfinite(arr)]
            else:
                arr = np.log10(fields[i])
                arr[idomain <= 0] = np.nan
                vals = arr[np.isfinite(arr)]
            vmins.append(np.percentile(vals, 2))
            vmaxs.append(np.percentile(vals, 98))
    for i in range(4):
        for j in range(3):
            ax = axes[i, j]
            if i == 3:
                if j != 0:
                    ax.axis("off")
                    continue
                field = np.log10(fields[i]).copy()
                field[idomain[0, :, :] <= 0] = np.nan
            else:
                field = np.log10(fields[i][j]).copy()
                field[idomain[j, :, :] <= 0] = np.nan
            im = ax.imshow(
                field,
                origin="upper",
                vmin=vmins[i],
                vmax=vmaxs[i],
                cmap="jet"
            )
            ax.tick_params(axis='both', which='major', labelsize=10)
            if i == 3:
                ax.set_title(param_names[i])
            else:
                ax.set_title(f"{param_names[i]}, L{j+1}")
                ax.set_xlabel("column", fontsize='11')
                ax.set_ylabel("row", fontsize='11')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad=0.05)
            fig.colorbar(im, cax=cax)


    axes[3, 1].axis("off")
    axes[3, 2].axis("off")
    return fig


def plot_fields_3l(fields, idomain, tag="hk", title="calibrated"):
    """Plot the 3 layers of one property (log10), inactive masked. (kept from v1)"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for k, ax in enumerate(axes):
        arr = fields[f"{tag}_l{k+1}"]
        z = np.log10(np.array(arr, float)); z[idomain[k] <= 0] = np.nan
        im = ax.imshow(z, cmap="viridis"); ax.set_title(f"{title} {tag.upper()} L{k+1} (log10)")
        plt.colorbar(im, ax=ax, shrink=0.6); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig

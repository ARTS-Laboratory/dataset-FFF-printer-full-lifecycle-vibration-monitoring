# -*- coding: utf-8 -*-
"""
plot_features.py
================

Generate Roberts-style stacked normalized-feature-trace figures for the
FFF printer wear monitoring DiB dataset.

Reads features.csv (produced by extract_features.py) and writes three
figures (one per sensor: Head, Motor, Frame).  Each figure has three
vertically stacked panels matching Downey's Chapter 3 categorisation:

    panel 1 : statistical    features (11 traces)
    panel 2 : time-series    features ( 8 traces)
    panel 3 : frequency-dom. features ( 8 traces)

Each trace is min-max scaled to [0, 1] within its own series and then
offset vertically so the traces stack without overlap.  The y-axis label
positions identify each trace by name.

NOTE: the within-trace min-max scaling is for *visual comparability only*.
The unscaled feature values are in features.csv and the values
normalized to wear state 1 are in features_normalized.csv.

Author:  Yanzhou Fu  (Florida Atlantic University)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

INPUT_CSV  = r"./output/features.csv"
OUTPUT_DIR = r"./output/figures"

CHANNELS = ["Head", "Motor", "Frame"]

# Vertical spacing between stacked traces (in min-max units).
# 1.0 is the tight bound; 1.15 leaves a small visual gap.
OFFSET_STEP = 1.15

# Save formats
SAVE_PDF = True
SAVE_PNG = True
DPI      = 300

# ------------------------------------------------------------
# Feature catalogue (must match extract_features.py)
# ------------------------------------------------------------
STATISTICAL_FEATURES = [
    "mean", "std", "var", "min", "max", "range",
    "median", "iqr", "mad", "skewness", "kurtosis",
]
TIME_SERIES_FEATURES = [
    "rms", "peak", "peak_to_peak", "abs_mean",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
]
FREQUENCY_FEATURES = [
    "peak_freq", "peak_ampl",
    "spectral_centroid", "spectral_spread",
    "spectral_skewness", "spectral_kurtosis",
    "band_power_0_100", "band_power_100_500",
]
CATEGORIES = [
    ("statistical",         STATISTICAL_FEATURES),
    ("time-series",         TIME_SERIES_FEATURES),
    ("frequency-domain",    FREQUENCY_FEATURES),
]

# ------------------------------------------------------------
# Pretty-print names for y-axis labels
# ------------------------------------------------------------
PRETTY_NAME = {
    "std":                "std. dev.",
    "var":                "variance",
    "iqr":                "IQR",
    "mad":                "mean abs. dev.",
    "rms":                "RMS",
    "peak_to_peak":       "peak-to-peak",
    "abs_mean":           "absolute mean",
    "crest_factor":       "crest factor",
    "shape_factor":       "shape factor",
    "impulse_factor":     "impulse factor",
    "clearance_factor":   "clearance factor",
    "peak_freq":          "peak freq.",
    "peak_ampl":          "peak ampl.",
    "spectral_centroid":  "spectral centroid",
    "spectral_spread":    "spectral spread",
    "spectral_skewness":  "spectral skewness",
    "spectral_kurtosis":  "spectral kurtosis",
    "band_power_0_100":   "band 0$-$100 Hz",
    "band_power_100_500": "band 100$-$500 Hz",
}


def prettify(name):
    """Return a clean display name for the y-axis label."""
    return PRETTY_NAME.get(name, name)


# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "serif"],
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   10,
    "mathtext.fontset":  "stix",
})


# ============================================================
# Per-trace scaling
# ============================================================

def minmax_scale(trace):
    """Min-max scale a 1-D trace to [0, 1].  Flat traces collapse to 0.5."""
    trace = np.asarray(trace, dtype=float)
    finite = np.isfinite(trace)
    if not np.any(finite):
        return np.full_like(trace, np.nan)
    lo = np.nanmin(trace)
    hi = np.nanmax(trace)
    rng = hi - lo
    if rng <= 0:
        return np.full_like(trace, 0.5)
    return (trace - lo) / rng


# ============================================================
# Single panel: stacked traces for one (sensor, category)
# ============================================================

def plot_panel(ax, df, channel, feature_names, offset_step=OFFSET_STEP):
    """
    Draw stacked min-max-scaled feature traces on `ax`.

    Bottom trace = first feature in `feature_names`.
    """
    wear_states = df["wear_state"].to_numpy()

    tick_positions = []
    tick_labels    = []
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, feat in enumerate(feature_names):
        col = f"{channel}_{feat}"
        if col not in df.columns:
            continue

        scaled = minmax_scale(df[col].to_numpy())
        offset = i * offset_step

        ax.plot(
            wear_states, scaled + offset,
            marker="o", markersize=4.5,
            linewidth=1.3,
            color=colors[i % 10],
        )

        tick_positions.append(offset + 0.5)
        tick_labels.append(prettify(feat))

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis="y", length=0)               # hide y-tick marks
    ax.set_xticks(wear_states)
    ax.set_xlim(wear_states.min() - 0.5, wear_states.max() + 0.5)
    ax.set_ylim(-0.2, (len(feature_names) - 1) * offset_step + 1.2)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ============================================================
# Full figure: three category panels for one sensor
# ============================================================

def make_figure_for_sensor(df, channel, out_dir):
    """One figure per sensor, with three vertically stacked category panels."""
    height_ratios = [len(feats) for _, feats in CATEGORIES]
    fig, axes = plt.subplots(
        nrows=len(CATEGORIES), ncols=1,
        figsize=(8.0, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.18},
    )

    for ax, (cat_name, feats) in zip(axes, CATEGORIES):
        plot_panel(ax, df, channel, feats)
        # Right-side annotation gives the panel its category label.
        ax.text(
            1.02, 0.5, cat_name,
            transform=ax.transAxes,
            rotation=270, va="center", ha="left",
            fontsize=11, fontstyle="italic",
        )

    axes[-1].set_xlabel("wear state")
    fig.suptitle(f"Normalized feature traces $-$ {channel} sensor",
                 fontsize=13, y=0.995)

    fig.tight_layout(rect=[0, 0, 0.97, 0.985])

    base = os.path.join(out_dir, f"feature_traces_{channel.lower()}")
    if SAVE_PDF:
        fig.savefig(base + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return base


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. "
            f"Run extract_features.py first to generate it."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # ---- Sanity check ----
    expected_total = len(CHANNELS) * sum(len(f) for _, f in CATEGORIES)
    meta = {"wear_state", "cumulative_hours"}
    feature_cols = [c for c in df.columns if c not in meta]
    if len(feature_cols) != expected_total:
        print(f"[warn] expected {expected_total} feature columns "
              f"(27 x {len(CHANNELS)}), found {len(feature_cols)}")

    # ---- Generate one figure per sensor ----
    for ch in CHANNELS:
        base = make_figure_for_sensor(df, ch, OUTPUT_DIR)
        suffixes = [".pdf"] * SAVE_PDF + [".png"] * SAVE_PNG
        for s in suffixes:
            print(f"[ok] saved: {base + s}")

    print()
    print(f"Figures complete.  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

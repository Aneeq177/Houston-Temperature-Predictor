"""
analyze_correlation.py
----------------------
Merges weather-station and surface-composition data, runs a full statistical
analysis of the relationship between impervious surface cover and land surface
temperature (Landsat 8, August 2023), and produces publication-ready figures.

Inputs
------
  data/weather_stations.csv    Station_ID | Latitude | Longitude | Avg_Summer_Temp (°C)
  data/surface_analysis.csv    Station_ID | Pct_Green | Pct_Impervious | Pct_Other | Method

Outputs  (all saved to results/)
-------
  scatter_impervious_vs_temp.png   Main result: impervious % → temperature
  scatter_green_vs_temp.png        Mirror: green space % → temperature
  correlation_heatmap.png          Full numeric correlation matrix
  residual_plot.png                OLS residuals — linearity & homoscedasticity check
  distributions.png                Marginal KDE + rug distributions of key variables
  joint_distribution.png           Joint scatter + marginals for the primary pair
  summary.txt                      Complete statistical report

Usage
-----
    python src/analyze_correlation.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # safe for headless / non-GUI environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_CSV  = PROJECT_ROOT / "data" / "weather_stations.csv"
SURFACE_CSV  = PROJECT_ROOT / "data" / "surface_analysis.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Global plot style  (publication-ready)
# ---------------------------------------------------------------------------
PALETTE = {
    "impervious": "#B5451B",   # brick red
    "green":      "#2C6E49",   # forest green
    "neutral":    "#4A4E69",   # muted blue-purple
    "accent":     "#F4A261",   # warm orange (CI bands, secondary elements)
}

plt.rcParams.update({
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     12,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.alpha":         0.4,
    "grid.linestyle":     "--",
})

sns.set_theme(style="whitegrid", rc=plt.rcParams)


# ===========================================================================
# Helpers
# ===========================================================================

def _fmt_p(p: float) -> str:
    """Format a p-value for display (e.g. '< 0.001' or '0.034')."""
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def _sig_stars(p: float) -> str:
    """Return significance stars for annotation (APA style)."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _save(fig: plt.Figure, filename: str) -> Path:
    out = RESULTS_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out.relative_to(PROJECT_ROOT)}")
    return out


# ===========================================================================
# 1. Data loading and merging
# ===========================================================================

def load_and_merge() -> pd.DataFrame:
    """Load both CSVs, inner-join on Station_ID, and validate."""
    for path in (WEATHER_CSV, SURFACE_CSV):
        if not path.exists():
            print(f"ERROR: Missing file — {path}", file=sys.stderr)
            print(
                "  Run the pipeline in order:\n"
                "    1. python src/fetch_weather_data.py\n"
                "    2. python src/fetch_satellite_images.py\n"
                "    3. python src/segment_surfaces.py\n"
                "    4. python src/analyze_correlation.py",
                file=sys.stderr,
            )
            sys.exit(1)

    weather = pd.read_csv(WEATHER_CSV)
    surface = pd.read_csv(SURFACE_CSV)

    n_before = len(weather)
    df = weather.merge(surface, on="Station_ID", how="inner")
    n_dropped = n_before - len(df)

    print(f"  Loaded {n_before} weather stations, {len(surface)} surface records.")
    print(f"  Merged → {len(df)} matched stations"
          + (f"  ({n_dropped} dropped — no surface analysis)" if n_dropped else "."))

    if len(df) < 5:
        print("WARNING: Fewer than 5 stations — statistical results may not be reliable.",
              file=sys.stderr)

    # Drop rows missing key numeric columns
    key_cols = ["Avg_Summer_Temp", "Pct_Impervious", "Pct_Green"]
    before = len(df)
    df = df.dropna(subset=key_cols)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} row(s) with missing values in key columns.")

    return df.reset_index(drop=True)


# ===========================================================================
# 2. Statistical analysis
# ===========================================================================

def run_statistics(df: pd.DataFrame) -> dict:
    """
    Run the full statistical battery and return results as a flat dict.
    Tests performed:
      - Pearson r       (Impervious → Temp, Green → Temp)
      - Spearman ρ      (same pairs — robust to outliers/non-normality)
      - Simple OLS      (Impervious → Temp)
      - Multiple OLS    (Impervious + Green → Temp)
      - Shapiro-Wilk    (normality of OLS residuals)
    """
    x_imp = df["Pct_Impervious"].values
    x_grn = df["Pct_Green"].values
    y_tmp = df["Avg_Summer_Temp"].values

    # ── Pearson ────────────────────────────────────────────────────────────
    pr_imp, pp_imp = stats.pearsonr(x_imp, y_tmp)
    pr_grn, pp_grn = stats.pearsonr(x_grn, y_tmp)

    # ── Spearman ───────────────────────────────────────────────────────────
    sr_imp, sp_imp = stats.spearmanr(x_imp, y_tmp)
    sr_grn, sp_grn = stats.spearmanr(x_grn, y_tmp)

    # ── Simple OLS: Temp ~ Impervious ──────────────────────────────────────
    slope, intercept, r_val, ols_p, stderr = stats.linregress(x_imp, y_tmp)
    ols_r2      = r_val ** 2
    fitted      = slope * x_imp + intercept
    residuals   = y_tmp - fitted

    # 95 % confidence interval on the slope
    n           = len(df)
    t_crit      = stats.t.ppf(0.975, df=n - 2)
    slope_ci    = t_crit * stderr           # ± this value

    # ── Multiple OLS: Temp ~ Impervious + Green ────────────────────────────
    X_multi  = df[["Pct_Impervious", "Pct_Green"]].values
    lr_multi = LinearRegression().fit(X_multi, y_tmp)
    r2_multi = lr_multi.score(X_multi, y_tmp)

    # ── Shapiro-Wilk normality test on residuals ───────────────────────────
    sw_stat, sw_p = stats.shapiro(residuals)

    return dict(
        n             = n,
        # Pearson
        pearson_r_imp = pr_imp,  pearson_p_imp = pp_imp,
        pearson_r_grn = pr_grn,  pearson_p_grn = pp_grn,
        # Spearman
        spear_r_imp   = sr_imp,  spear_p_imp   = sp_imp,
        spear_r_grn   = sr_grn,  spear_p_grn   = sp_grn,
        # Simple OLS
        ols_slope     = slope,   ols_intercept = intercept,
        ols_r2        = ols_r2,  ols_p         = ols_p,
        ols_stderr    = stderr,  slope_ci      = slope_ci,
        fitted        = fitted,  residuals     = residuals,
        # Multiple OLS
        r2_multi      = r2_multi,
        # Normality
        sw_stat       = sw_stat, sw_p          = sw_p,
        # Raw arrays (for plots)
        x_imp         = x_imp,
        x_grn         = x_grn,
        y_tmp         = y_tmp,
    )


# ===========================================================================
# 3. Visualisations
# ===========================================================================

# ── 3a. Scatter + OLS trendline ────────────────────────────────────────────

def _scatter_with_trend(
    df: pd.DataFrame,
    st: dict,
    x_col: str,
    x_label: str,
    color: str,
    filename: str,
    pearson_r: float,
    pearson_p: float,
    spear_r: float,
) -> None:
    """Generic scatter + seaborn regplot + stats annotation."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Points coloured by the complementary variable
    complement = "Pct_Green" if x_col == "Pct_Impervious" else "Pct_Impervious"
    cmap_name  = "RdYlGn_r" if x_col == "Pct_Impervious" else "RdYlGn"
    sc = ax.scatter(
        df[x_col], df["Avg_Summer_Temp"],
        c=df[complement], cmap=cmap_name,
        s=90, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=4,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(
        "% Green Space" if complement == "Pct_Green" else "% Impervious",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

    # OLS regression line + 95 % confidence band (seaborn handles both)
    sns.regplot(
        data=df, x=x_col, y="Avg_Summer_Temp",
        scatter=False, ci=95,
        line_kws={"color": color, "linewidth": 2.0, "label": "OLS fit"},
        ax=ax,
    )

    # Stats annotation box
    r2   = pearson_r ** 2
    text = (
        f"Pearson  r = {pearson_r:+.3f}{_sig_stars(pearson_p)}\n"
        f"Spearman ρ = {spear_r:+.3f}\n"
        f"R²          = {r2:.3f}\n"
        f"p-value     = {_fmt_p(pearson_p)}"
    )
    ax.text(
        0.04, 0.97, text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Land Surface Temp. (°C)")
    title_prefix = "Urban Heat Island" if x_col == "Pct_Impervious" else "Vegetation Cooling"
    ax.set_title(f"{title_prefix}: {x_label} vs. Land Surface Temp.\n"
                 f"Houston Metro Stations  (n = {st['n']},  August 2023 — Landsat 8)")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.35)

    _save(fig, filename)


def plot_scatter_impervious(df: pd.DataFrame, st: dict) -> None:
    _scatter_with_trend(
        df, st,
        x_col="Pct_Impervious",
        x_label="Impervious Surface Cover (%)",
        color=PALETTE["impervious"],
        filename="scatter_impervious_vs_temp.png",
        pearson_r=st["pearson_r_imp"],
        pearson_p=st["pearson_p_imp"],
        spear_r=st["spear_r_imp"],
    )


def plot_scatter_green(df: pd.DataFrame, st: dict) -> None:
    _scatter_with_trend(
        df, st,
        x_col="Pct_Green",
        x_label="Green Space Cover (%)",
        color=PALETTE["green"],
        filename="scatter_green_vs_temp.png",
        pearson_r=st["pearson_r_grn"],
        pearson_p=st["pearson_p_grn"],
        spear_r=st["spear_r_grn"],
    )


# ── 3b. Correlation heatmap ────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = [
        "Avg_Summer_Temp", "Pct_Impervious", "Pct_Green",
        "Pct_Other", "Latitude", "Longitude",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available].corr(method="pearson")

    # Friendly axis labels
    labels = {
        "Avg_Summer_Temp": "Land Surface\nTemp (°C)",
        "Pct_Impervious":  "Impervious\nSurface (%)",
        "Pct_Green":       "Green\nSpace (%)",
        "Pct_Other":       "Other\nSurface (%)",
        "Latitude":        "Latitude",
        "Longitude":       "Longitude",
    }
    renamed = corr_matrix.rename(index=labels, columns=labels)

    n = len(available)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))

    mask = np.triu(np.ones_like(renamed, dtype=bool), k=1)   # hide upper triangle
    sns.heatmap(
        renamed,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#e0e0e0",
        square=True, ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title(
        "Pearson Correlation Matrix — Houston Station Variables\n"
        "(lower triangle shown)",
        pad=14,
    )
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    _save(fig, "correlation_heatmap.png")


# ── 3c. OLS residual plot ──────────────────────────────────────────────────

def plot_residuals(df: pd.DataFrame, st: dict) -> None:
    fitted    = st["fitted"]
    residuals = st["residuals"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Residuals vs Fitted
    ax = axes[0]
    ax.scatter(fitted, residuals,
               color=PALETTE["neutral"], s=70, alpha=0.8,
               edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=PALETTE["impervious"], linewidth=1.5, linestyle="--")
    # Lowess smoother to reveal any non-linear trend
    from statsmodels.nonparametric.smoothers_lowess import lowess  # soft import
    lw = lowess(residuals, fitted, frac=0.5)
    ax.plot(lw[:, 0], lw[:, 1], color=PALETTE["accent"],
            linewidth=1.5, linestyle="-", label="LOWESS")
    ax.set_xlabel("Fitted Values (°C)")
    ax.set_ylabel("Residuals (°C)")
    ax.set_title("Residuals vs. Fitted Values\n(check: should scatter randomly around 0)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    # Panel 2: Q-Q plot (normality of residuals)
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, color=PALETTE["neutral"], s=70, alpha=0.8,
                edgecolors="white", linewidths=0.5, label="Residuals", zorder=4)
    x_line = np.array([min(osm), max(osm)])
    ax2.plot(x_line, slope * x_line + intercept,
             color=PALETTE["impervious"], linewidth=1.8, linestyle="--",
             label="Normal reference")
    sw_text = (f"Shapiro-Wilk: W = {st['sw_stat']:.3f},  "
               f"p = {_fmt_p(st['sw_p'])}")
    ax2.text(0.05, 0.95, sw_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles (Residuals °C)")
    ax2.set_title("Normal Q–Q Plot of OLS Residuals\n(check: points should follow the line)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)

    fig.suptitle("OLS Regression Diagnostics  —  Temp ~ Impervious Surface",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "residual_plot.png")


# ── 3d. Marginal distributions ────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, color, label in [
        (axes[0], "Pct_Impervious", PALETTE["impervious"], "Impervious Surface Cover (%)"),
        (axes[1], "Avg_Summer_Temp", PALETTE["neutral"],   "Land Surface Temp. (°C)"),
    ]:
        sns.histplot(df[col], kde=True, color=color, alpha=0.45, ax=ax,
                     edgecolor="white", linewidth=0.4)
        ax.axvline(df[col].mean(), color=color, linewidth=2,
                   linestyle="--", label=f"Mean = {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color=color, linewidth=2,
                   linestyle=":", label=f"Median = {df[col].median():.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

    fig.suptitle("Marginal Distributions — Key Study Variables",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "distributions.png")


# ── 3e. Joint distribution ─────────────────────────────────────────────────

def plot_joint_distribution(df: pd.DataFrame, st: dict) -> None:
    g = sns.JointGrid(
        data=df,
        x="Pct_Impervious",
        y="Avg_Summer_Temp",
        height=7,
        ratio=4,
    )

    # Central scatter + regression line
    g.plot_joint(
        sns.regplot,
        color=PALETTE["impervious"],
        scatter_kws=dict(s=80, alpha=0.75, edgecolors="white", linewidths=0.5),
        line_kws=dict(linewidth=2),
        ci=95,
    )

    # Marginal KDE plots
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.4, color=PALETTE["neutral"])

    g.set_axis_labels(
        "Impervious Surface Cover (%)",
        "Land Surface Temp. (°C)",
        fontsize=12, fontweight="bold",
    )

    r2    = st["pearson_r_imp"] ** 2
    title = (
        f"Joint Distribution — Impervious Cover vs. Land Surface Temperature\n"
        f"r = {st['pearson_r_imp']:+.3f}   R² = {r2:.3f}   "
        f"p = {_fmt_p(st['pearson_p_imp'])}   n = {st['n']}"
    )
    g.figure.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "joint_distribution.png"
    g.figure.savefig(out, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  Saved → {out.relative_to(PROJECT_ROOT)}")


# ===========================================================================
# 4. Text summary
# ===========================================================================

def build_summary(df: pd.DataFrame, st: dict) -> str:
    """Return a formatted statistical summary string."""
    imp_desc  = df["Pct_Impervious"].describe()
    grn_desc  = df["Pct_Green"].describe()
    temp_desc = df["Avg_Summer_Temp"].describe()

    def sig_note(p):
        return "significant ✓" if p < 0.05 else "not significant"

    r2_imp = st["pearson_r_imp"] ** 2
    r2_grn = st["pearson_r_grn"] ** 2

    lines = [
        "=" * 66,
        "  HOUSTON URBAN HEAT ISLAND — STATISTICAL ANALYSIS SUMMARY",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 66,
        "",
        "  DATASET",
        f"    Stations analysed         : {st['n']}",
        f"    LST range (Aug 2023)      : {temp_desc['min']:.2f} – {temp_desc['max']:.2f} °C "
        f"  (mean {temp_desc['mean']:.2f} °C)",
        f"    Impervious cover range    : {imp_desc['min']:.1f} – {imp_desc['max']:.1f} %"
        f"  (mean {imp_desc['mean']:.1f} %)",
        f"    Green space range         : {grn_desc['min']:.1f} – {grn_desc['max']:.1f} %"
        f"  (mean {grn_desc['mean']:.1f} %)",
        "",
        "─" * 66,
        "  PEARSON CORRELATION  (linear association)",
        "─" * 66,
        f"    Impervious → Temp :  r = {st['pearson_r_imp']:+.4f}   "
        f"p = {_fmt_p(st['pearson_p_imp'])}   R² = {r2_imp:.4f}   "
        f"[{sig_note(st['pearson_p_imp'])}]",
        f"    Green Space → Temp:  r = {st['pearson_r_grn']:+.4f}   "
        f"p = {_fmt_p(st['pearson_p_grn'])}   R² = {r2_grn:.4f}   "
        f"[{sig_note(st['pearson_p_grn'])}]",
        "",
        "─" * 66,
        "  SPEARMAN CORRELATION  (rank-based, robust to outliers)",
        "─" * 66,
        f"    Impervious → Temp :  ρ = {st['spear_r_imp']:+.4f}   "
        f"p = {_fmt_p(st['spear_p_imp'])}   [{sig_note(st['spear_p_imp'])}]",
        f"    Green Space → Temp:  ρ = {st['spear_r_grn']:+.4f}   "
        f"p = {_fmt_p(st['spear_p_grn'])}   [{sig_note(st['spear_p_grn'])}]",
        "",
        "─" * 66,
        "  SIMPLE OLS REGRESSION  —  LST ~ Pct_Impervious",
        "─" * 66,
        f"    Slope             : {st['ols_slope']:+.4f} °C per 1% impervious cover",
        f"    95% CI on slope   : ± {st['slope_ci']:.4f} °C",
        f"    Intercept         : {st['ols_intercept']:.4f} °C",
        f"    R²                : {st['ols_r2']:.4f}  "
        f"(explains {st['ols_r2']*100:.1f}% of temperature variance)",
        f"    p-value           : {_fmt_p(st['ols_p'])}   [{sig_note(st['ols_p'])}]",
        f"    Std. Error        : {st['ols_stderr']:.4f}",
        "",
        "─" * 66,
        "  MULTIPLE OLS REGRESSION  —  LST ~ Pct_Impervious + Pct_Green",
        "─" * 66,
        f"    R²                : {st['r2_multi']:.4f}  "
        f"(explains {st['r2_multi']*100:.1f}% of temperature variance)",
        f"    Marginal gain vs simple OLS: "
        f"{(st['r2_multi'] - st['ols_r2'])*100:+.1f} pp",
        "",
        "─" * 66,
        "  NORMALITY OF RESIDUALS  (Shapiro-Wilk)",
        "─" * 66,
        f"    W = {st['sw_stat']:.4f}   p = {_fmt_p(st['sw_p'])}",
        "    Residuals appear normally distributed (p > 0.05 → Pearson is appropriate)."
        if st["sw_p"] > 0.05 else
        "    Residuals deviate from normality (p ≤ 0.05 → prefer Spearman interpretation).",
        "",
        "─" * 66,
        "  INTERPRETATION",
        "─" * 66,
    ]

    # Contextual interpretation block
    direction_imp = "positive" if st["pearson_r_imp"] > 0 else "negative"
    direction_grn = "positive" if st["pearson_r_grn"] > 0 else "negative"
    magnitude_imp = abs(st["pearson_r_imp"])

    if magnitude_imp < 0.3:
        strength = "weak"
    elif magnitude_imp < 0.6:
        strength = "moderate"
    else:
        strength = "strong"

    interp = f"""\
    There is a {strength} {direction_imp} correlation between impervious surface
    cover and land surface temperature (LST) across Houston metro stations
    (Pearson r = {st['pearson_r_imp']:+.3f}, p = {_fmt_p(st['pearson_p_imp'])}).
    LST is derived from Landsat 8 30 m thermal imagery (August 2023).

    The OLS regression estimates that each additional 1% of impervious
    cover is associated with a {abs(st['ols_slope']):.3f} °C {'increase' if st['ols_slope'] > 0 else 'decrease'}
    in LST (95% CI: ±{st['slope_ci']:.3f} °C per %).

    Green space shows a {direction_grn} association with LST
    (r = {st['pearson_r_grn']:+.3f}, p = {_fmt_p(st['pearson_p_grn'])}), consistent with
    the urban heat-island hypothesis that vegetation provides a cooling effect.

    The simple model (Impervious only) explains {st['ols_r2']*100:.1f}% of temperature
    variance; adding Green Space raises this to {st['r2_multi']*100:.1f}%.
    Note: with n = {st['n']} stations, results should be interpreted cautiously.
    Increasing spatial coverage would strengthen statistical power."""

    lines.append(textwrap.dedent(interp))
    lines.append("")
    lines.append("=" * 66)
    return "\n".join(lines)


def print_and_save_summary(df: pd.DataFrame, st: dict) -> None:
    summary = build_summary(df, st)
    print("\n" + summary)
    out = RESULTS_DIR / "summary.txt"
    out.write_text(summary, encoding="utf-8")
    print(f"\n  Summary also saved → {out.relative_to(PROJECT_ROOT)}")


# ===========================================================================
# 5. Entry point
# ===========================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  Houston Surface–Temperature Correlation Analysis")
    print("=" * 66)

    # ── Load ───────────────────────────────────────────────────────────────
    print("\n[1/4] Loading and merging data …")
    df = load_and_merge()

    # ── Statistics ─────────────────────────────────────────────────────────
    print("\n[2/4] Running statistical tests …")
    st = run_statistics(df)

    # ── Visualisations ─────────────────────────────────────────────────────
    print(f"\n[3/4] Generating figures → results/")
    plot_scatter_impervious(df, st)
    plot_scatter_green(df, st)
    plot_correlation_heatmap(df)
    try:
        plot_residuals(df, st)
    except ImportError:
        # statsmodels LOWESS is optional; fall back to plain residual plot
        print("  [INFO] statsmodels not installed — skipping LOWESS on residual plot.")
        _plot_residuals_no_lowess(df, st)
    plot_distributions(df)
    plot_joint_distribution(df, st)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[4/4] Statistical summary\n")
    print_and_save_summary(df, st)

    print("\nAll outputs written to results/\nDone.")


def _plot_residuals_no_lowess(df: pd.DataFrame, st: dict) -> None:
    """Residual plot without the optional statsmodels LOWESS smoother."""
    fitted    = st["fitted"]
    residuals = st["residuals"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(fitted, residuals, color=PALETTE["neutral"], s=70, alpha=0.8,
               edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=PALETTE["impervious"], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Fitted Values (°C)")
    ax.set_ylabel("Residuals (°C)")
    ax.set_title("Residuals vs. Fitted Values")
    ax.grid(True, alpha=0.35)

    ax2 = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, color=PALETTE["neutral"], s=70, alpha=0.8,
                edgecolors="white", linewidths=0.5)
    x_line = np.array([min(osm), max(osm)])
    ax2.plot(x_line, slope * x_line + intercept,
             color=PALETTE["impervious"], linewidth=1.8, linestyle="--")
    ax2.text(0.05, 0.95,
             f"Shapiro-Wilk: W={st['sw_stat']:.3f}, p={_fmt_p(st['sw_p'])}",
             transform=ax2.transAxes, fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles (Residuals °C)")
    ax2.set_title("Normal Q–Q Plot of OLS Residuals")
    ax2.grid(True, alpha=0.35)

    fig.suptitle("OLS Regression Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "residual_plot.png")


if __name__ == "__main__":
    main()

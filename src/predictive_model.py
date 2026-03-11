"""
predictive_model.py
-------------------
Trains machine learning models (Linear Regression, Random Forest, and
Gradient Boosting) to predict land surface temperature from surface
composition and geographic features.

Features
--------
  Surface composition : Pct_Green, Pct_Impervious, Pct_Other
  Geographic          : Latitude, Longitude, Dist_Coast_km, Dist_Center_km

  Note: Linear Regression omits Pct_Other to avoid perfect multicollinearity
        (Green + Impervious + Other = 100%).  Tree models use all 7 features.

  Dist_Coast_km  — great-circle distance to Galveston Bay / Gulf coast
                   (major cooling factor for southern Houston stations)
  Dist_Center_km — great-circle distance to Houston CBD
                   (proxy for urban heat island intensity)

Simulator model strategy
------------------------
  In-range inputs  → Gradient Boosting (more accurate within training data)
  Out-of-range     → Linear Regression  (can extrapolate; GB cannot)

Phases
------
  2 — Model training & cross-validation
  3 — Ground truth validation (CV predictions vs actual, saves scatter plot)
  4 — Interactive 'What-If' microclimate simulator

Usage
-----
    python src/predictive_model.py
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict, KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_CSV  = PROJECT_ROOT / "data" / "weather_stations.csv"
SURFACE_CSV  = PROJECT_ROOT / "data" / "surface_analysis.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Geographic reference points (Houston area) ───────────────────────────────
COAST_LAT,    COAST_LON    = 29.20, -94.85   # Galveston Bay / Gulf coast
DOWNTOWN_LAT, DOWNTOWN_LON = 29.76, -95.37   # Houston CBD

# ── Feature sets ─────────────────────────────────────────────────────────────
# LR omits Pct_Other — Green + Impervious + Other = 100% causes singular matrix
LR_FEATURES   = [
    "Pct_Green", "Pct_Impervious",
    "Latitude", "Longitude",
    "Dist_Coast_km", "Dist_Center_km",
]
TREE_FEATURES = [
    "Pct_Green", "Pct_Impervious", "Pct_Other",
    "Latitude", "Longitude",
    "Dist_Coast_km", "Dist_Center_km",
]

CV_SPLITS = 5


# ── Haversine helper ─────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R    = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


# ── Data loading & feature engineering ──────────────────────────────────────

def load_and_merge_data() -> pd.DataFrame:
    """Loads, merges, and feature-engineers the weather and surface data."""
    if not WEATHER_CSV.exists() or not SURFACE_CSV.exists():
        print("Error: Missing data files. Make sure CV pipeline has run.")
        sys.exit(1)

    weather = pd.read_csv(WEATHER_CSV)
    surface = pd.read_csv(SURFACE_CSV)

    df = weather.merge(surface, on="Station_ID", how="inner")

    # Ensure Pct_Other exists (older CSVs may not have it)
    if "Pct_Other" not in df.columns:
        df["Pct_Other"] = 100.0 - df["Pct_Green"] - df["Pct_Impervious"]

    # ── Derived geographic features ──────────────────────────────────────────
    df["Dist_Coast_km"]  = df.apply(
        lambda r: haversine_km(r["Latitude"], r["Longitude"], COAST_LAT, COAST_LON),
        axis=1,
    )
    df["Dist_Center_km"] = df.apply(
        lambda r: haversine_km(r["Latitude"], r["Longitude"], DOWNTOWN_LAT, DOWNTOWN_LON),
        axis=1,
    )

    # ── Drop any row that has a NaN in any feature or target column ──────────
    # Landsat 8 LST can return None for cloudy stations; Sentinel-2 chips can
    # occasionally produce NaN surface percentages. Drop all affected rows here
    # so sklearn never sees NaN values during training.
    required_cols = ["Avg_Summer_Temp"] + [c for c in LR_FEATURES + TREE_FEATURES
                                            if c in df.columns]
    before = len(df)
    df = df.dropna(subset=list(dict.fromkeys(required_cols)))  # deduplicated list
    dropped = before - len(df)
    if dropped:
        print(f"  [load_and_merge_data] Dropped {dropped} row(s) with NaN values "
              f"({before} → {len(df)} stations).")

    if len(df) < CV_SPLITS:
        print(f"Error: Only {len(df)} clean stations remain — need at least {CV_SPLITS}.")
        sys.exit(1)

    return df


# ── Training & evaluation ────────────────────────────────────────────────────

def train_and_evaluate_models(df: pd.DataFrame):
    """
    Trains and evaluates three models using 5-fold cross-validation, then
    fits all three on the full dataset for downstream use.

    Returns
    -------
    lr_model : LinearRegression        — for out-of-range extrapolation
    X_lr     : pd.DataFrame            — LR feature matrix (6 features)
    gb_model : GradientBoostingRegressor — for in-range predictions
    X_tree   : pd.DataFrame            — tree feature matrix (7 features)
    df       : pd.DataFrame            — processed dataframe
    """
    print("=" * 60)
    print("  PHASE 2: PREDICTIVE MODEL TRAINING")
    print("=" * 60)

    y = df["Avg_Summer_Temp"]

    X_lr   = df[[c for c in LR_FEATURES   if c in df.columns]]
    X_tree = df[[c for c in TREE_FEATURES if c in df.columns]]

    print(f"\n  LR features   ({len(X_lr.columns)}): {list(X_lr.columns)}")
    print(f"  Tree features ({len(X_tree.columns)}): {list(X_tree.columns)}")
    print(f"  Stations: {len(df)}")

    kf         = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    cv_scoring = {"r2": "r2", "mse": "neg_mean_squared_error"}

    # ── Model 1: Linear Regression ───────────────────────────────────────────
    lr_model = LinearRegression()
    lr_cv    = cross_validate(lr_model, X_lr, y, cv=kf, scoring=cv_scoring)
    lr_r2    = lr_cv["test_r2"]
    lr_rmse  = np.sqrt(-lr_cv["test_mse"])

    print("\n--- Model: Linear Regression (baseline) ---")
    print(f"  CV R²  : {lr_r2.mean():.3f}  (±{lr_r2.std():.3f})")
    print(f"  CV RMSE: {lr_rmse.mean():.3f} °C  (±{lr_rmse.std():.3f})")

    # ── Model 2: Random Forest ───────────────────────────────────────────────
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_cv    = cross_validate(rf_model, X_tree, y, cv=kf, scoring=cv_scoring)
    rf_r2    = rf_cv["test_r2"]
    rf_rmse  = np.sqrt(-rf_cv["test_mse"])

    print("\n--- Model: Random Forest ---")
    print(f"  CV R²  : {rf_r2.mean():.3f}  (±{rf_r2.std():.3f})")
    print(f"  CV RMSE: {rf_rmse.mean():.3f} °C  (±{rf_rmse.std():.3f})")

    # ── Model 3: Gradient Boosting ───────────────────────────────────────────
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    gb_cv   = cross_validate(gb_model, X_tree, y, cv=kf, scoring=cv_scoring)
    gb_r2   = gb_cv["test_r2"]
    gb_rmse = np.sqrt(-gb_cv["test_mse"])

    print("\n--- Model: Gradient Boosting ---")
    print(f"  CV R²  : {gb_r2.mean():.3f}  (±{gb_r2.std():.3f})")
    print(f"  CV RMSE: {gb_rmse.mean():.3f} °C  (±{gb_rmse.std():.3f})")

    # ── Winner ───────────────────────────────────────────────────────────────
    scores = {
        "Linear Regression": lr_r2.mean(),
        "Random Forest":     rf_r2.mean(),
        "Gradient Boosting": gb_r2.mean(),
    }
    winner = max(scores, key=lambda k: scores[k])
    print("\n--- Model Comparison ---")
    for name, r2 in scores.items():
        marker = "  <- best" if name == winner else ""
        print(f"  {name:22}: CV R² = {r2:.3f}{marker}")

    # ── Fit all models on full data ──────────────────────────────────────────
    lr_model.fit(X_lr, y)
    rf_model.fit(X_tree, y)
    gb_model.fit(X_tree, y)

    print("\n--- Linear Regression Coefficients ---")
    dist_features = {"Dist_Coast_km", "Dist_Center_km"}
    geo_features  = {"Latitude", "Longitude"}
    for feat, coef in zip(X_lr.columns, lr_model.coef_):
        if feat in dist_features:
            unit = "°C per km"
        elif feat in geo_features:
            unit = "°C per 1°"
        else:
            unit = "°C per 1%"
        print(f"  {feat:22}: {coef:+.4f} {unit}")
    print(f"  {'Intercept':22}: {lr_model.intercept_:.4f} °C")

    print("\n--- Feature Importance (Gradient Boosting) ---")
    for feat, imp in sorted(
        zip(X_tree.columns, gb_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    ):
        bar = "█" * int(imp * 40)
        print(f"  {feat:22}: {imp:.1%}  {bar}")

    return lr_model, X_lr, gb_model, X_tree, df


# ── Ground truth validation ──────────────────────────────────────────────────

def validate_predictions(
    gb_model: GradientBoostingRegressor,
    X_tree: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """
    Validates predictions against ground truth using cross-validated predictions.

    Uses cross_val_predict so every station gets a held-out prediction (it was
    never seen during its fold's training).  Prints a per-station table and
    saves a predicted-vs-actual scatter plot to results/prediction_vs_actual.png.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: GROUND TRUTH VALIDATION")
    print("=" * 60)

    y = df["Avg_Summer_Temp"]

    # Fresh GB instance — cross_val_predict must train from scratch per fold
    gb_fresh = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    kf       = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(gb_fresh, X_tree, y, cv=kf)

    residuals = y.values - cv_preds
    rmse      = float(np.sqrt((residuals ** 2).mean()))
    mae       = float(np.abs(residuals).mean())
    ss_res    = float((residuals ** 2).sum())
    ss_tot    = float(((y.values - y.mean()) ** 2).sum())
    r2        = 1.0 - ss_res / ss_tot
    max_err   = float(np.abs(residuals).max())

    print(f"\n  Cross-validated performance (Gradient Boosting, {CV_SPLITS}-fold):")
    print(f"    CV R²     : {r2:.3f}  (1.0 = perfect)")
    print(f"    CV RMSE   : {rmse:.3f} °C")
    print(f"    CV MAE    : {mae:.3f} °C")
    print(f"    Max error : {max_err:.2f} °C")

    print(f"\n  {'Station ID':30} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print(f"  {'-' * 62}")
    for sid, actual, pred in zip(df["Station_ID"].values, y.values, cv_preds):
        err  = pred - actual
        flag = "  <- outlier" if abs(err) > 2 * rmse else ""
        print(f"  {str(sid):30} {actual:>8.2f} {pred:>10.2f} {err:>+8.2f}{flag}")

    # ── Scatter plot ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        y, cv_preds,
        c=residuals, cmap="RdYlGn_r", s=70,
        edgecolors="k", linewidths=0.5, alpha=0.85,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="Residual  (actual − predicted, °C)")

    lims = [min(y.min(), cv_preds.min()) - 1.5,
            max(y.max(), cv_preds.max()) + 1.5]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction", zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Actual LST (°C)", fontsize=12)
    ax.set_ylabel("Predicted LST (°C)", fontsize=12)
    ax.set_title(
        f"Predicted vs. Actual LST — Gradient Boosting ({CV_SPLITS}-fold CV)\n"
        f"CV R² = {r2:.3f}   RMSE = {rmse:.3f} °C   MAE = {mae:.3f} °C",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = RESULTS_DIR / "prediction_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved scatter plot -> results/prediction_vs_actual.png")


# ── What-If simulator ────────────────────────────────────────────────────────

def run_simulator(
    lr_model: LinearRegression,
    X_lr: pd.DataFrame,
    gb_model: GradientBoostingRegressor,
    X_tree: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """
    Interactive 'What-If' microclimate simulator.

    Allows the user to select a specific anchor station, the city average,
    or enter custom coordinates with baseline ground truth data. 
    It then predicts the temperature difference based on simulated surface modifications.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: 'WHAT-IF' MICROCLIMATE SIMULATOR")
    print("  (GB for in-range inputs, LR for extrapolation)")
    print("=" * 60)

    # ── Training bounds for extrapolation check ────────────────────────────
    green_min, green_max = X_tree["Pct_Green"].min(),      X_tree["Pct_Green"].max()
    imp_min,   imp_max   = X_tree["Pct_Impervious"].min(), X_tree["Pct_Impervious"].max()

    city_avg_temp = df["Avg_Summer_Temp"].mean()
    city_avg_grn  = df["Pct_Green"].mean()
    city_avg_imp  = df["Pct_Impervious"].mean()

    print(f"  Training surface ranges:")
    print(f"    Pct_Green      : {green_min:.1f}% – {green_max:.1f}%")
    print(f"    Pct_Impervious : {imp_min:.1f}% – {imp_max:.1f}%")
    print("-" * 60)

    # Need access to global anchors for custom coordinates
    global COAST_LAT, COAST_LON, DOWNTOWN_LAT, DOWNTOWN_LON

    # ── Known station IDs for the help prompt ─────────────────────────────
    station_ids: list[str] = sorted(str(s) for s in df["Station_ID"])

    while True:
        try:
            print("\n--- NEW SIMULATION ---")
            sample = ", ".join(s for i, s in enumerate(station_ids) if i < 5)
            print(f"  Known station IDs (sample): {sample} ...")
            station_input = input(
                "  Enter Station_ID, 'custom' for custom coordinates, or blank to exit: "
            ).strip()
            if not station_input:
                break

            # ── Location selection ────────────────────────────────────────
            if station_input.lower() == "custom":
                print("\n  Enter current ground-truth data for your location:")
                anchor_lat    = float(input("    Latitude          (e.g., 29.76) : "))
                anchor_lon    = float(input("    Longitude         (e.g., -95.37): "))
                current_temp  = float(input("    True temperature  (°C)          : "))
                current_grn   = float(input("    Current % Green Space           : "))
                current_imp   = float(input("    Current % Impervious Surface    : "))
                anchor_coast  = haversine_km(anchor_lat, anchor_lon, COAST_LAT,    COAST_LON)
                anchor_center = haversine_km(anchor_lat, anchor_lon, DOWNTOWN_LAT, DOWNTOWN_LON)
                location_name = f"Custom ({anchor_lat:.4f}°N, {anchor_lon:.4f}°W)"
            else:
                matches = df[df["Station_ID"].str.contains(station_input, case=False, na=False)]
                if len(matches) == 0:
                    print(f"\n  Station '{station_input}' not found.")
                    print(f"  All station IDs: {', '.join(station_ids)}")
                    continue
                if len(matches) > 1:
                    print(f"  Multiple matches — using first: {matches.iloc[0]['Station_ID']}")
                match         = matches.iloc[0]
                anchor_lat    = float(match["Latitude"])
                anchor_lon    = float(match["Longitude"])
                anchor_coast  = float(match["Dist_Coast_km"])
                anchor_center = float(match["Dist_Center_km"])
                current_temp  = float(match["Avg_Summer_Temp"])
                current_grn   = float(match["Pct_Green"])
                current_imp   = float(match["Pct_Impervious"])
                location_name = str(match["Station_ID"])

            print(f"\n  Location  : {location_name}")
            print(f"  Coords    : {anchor_lat:.4f}°N, {anchor_lon:.4f}°W")
            print(f"  Dist coast: {anchor_coast:.1f} km   |   Dist CBD: {anchor_center:.1f} km")
            print(f"  Current   : Temp {current_temp:.2f} °C  |  "
                  f"Green {current_grn:.1f}%  |  Impervious {current_imp:.1f}%  |  "
                  f"Other {100 - current_grn - current_imp:.1f}%")

            # ── What-If inputs ────────────────────────────────────────────
            print("\n  Enter your hypothetical scenario (blank = keep current value):")
            green_raw = input(f"    New % Green Space  [current: {current_grn:.1f}%]: ").strip()
            imp_raw   = input(f"    New % Impervious   [current: {current_imp:.1f}%]: ").strip()

            new_green = float(green_raw) if green_raw else current_grn
            new_imp   = float(imp_raw)   if imp_raw   else current_imp

            if not (0.0 <= new_green <= 100.0) or not (0.0 <= new_imp <= 100.0):
                print("  ERROR: Percentages must be between 0 and 100.")
                continue
            if new_green + new_imp > 100.0:
                print(f"  ERROR: Green ({new_green:.1f}%) + Impervious ({new_imp:.1f}%) "
                      f"= {new_green + new_imp:.1f}% — exceeds 100%.")
                continue

            new_other  = 100.0 - new_green - new_imp
            curr_other = 100.0 - current_grn - current_imp

            # ── Inner helper: build a prediction for one surface state ────
            def predict_state(grn: float, imp: float, oth: float) -> tuple[float, str]:
                out_of_range = (
                    grn < green_min or grn > green_max or
                    imp < imp_min   or imp > imp_max
                )
                if out_of_range:
                    row_lr: dict[str, list] = {col: [0.0] for col in X_lr.columns}
                    row_lr["Pct_Green"]      = [grn]
                    row_lr["Pct_Impervious"] = [imp]
                    row_lr["Latitude"]       = [anchor_lat]
                    row_lr["Longitude"]      = [anchor_lon]
                    if "Dist_Coast_km"  in row_lr: row_lr["Dist_Coast_km"]  = [anchor_coast]
                    if "Dist_Center_km" in row_lr: row_lr["Dist_Center_km"] = [anchor_center]
                    X_df = pd.DataFrame(row_lr)[list(X_lr.columns)]
                    return float(lr_model.predict(X_df)[0]), "Linear Regression (extrapolation)"
                else:
                    row_gb: dict[str, list] = {col: [0.0] for col in X_tree.columns}
                    row_gb["Pct_Green"]      = [grn]
                    row_gb["Pct_Impervious"] = [imp]
                    row_gb["Latitude"]       = [anchor_lat]
                    row_gb["Longitude"]      = [anchor_lon]
                    if "Dist_Coast_km"  in row_gb: row_gb["Dist_Coast_km"]  = [anchor_coast]
                    if "Dist_Center_km" in row_gb: row_gb["Dist_Center_km"] = [anchor_center]
                    if "Pct_Other"      in row_gb: row_gb["Pct_Other"]      = [oth]
                    X_df = pd.DataFrame(row_gb)[list(X_tree.columns)]
                    return float(gb_model.predict(X_df)[0]), "Gradient Boosting"

            # ── Predict current and new states ────────────────────────────
            pred_curr, _          = predict_state(current_grn, current_imp, curr_other)
            pred_new,  model_used = predict_state(new_green,   new_imp,     new_other)

            delta               = pred_new - pred_curr   # model-estimated change
            estimated_real_temp = current_temp + delta   # applied to true baseline

            # ── Output ───────────────────────────────────────────────────
            print("\n" + "─" * 60)
            print(f"  RESULTS FOR: {location_name}")
            print("─" * 60)
            print(f"  Surface change:")
            print(f"    Green Space  : {current_grn:.1f}%  →  {new_green:.1f}%  "
                  f"({new_green - current_grn:+.1f}%)")
            print(f"    Impervious   : {current_imp:.1f}%  →  {new_imp:.1f}%  "
                  f"({new_imp - current_imp:+.1f}%)")
            print(f"    Other        : {curr_other:.1f}%  →  {new_other:.1f}%")
            print()
            print(f"  Model baseline (current surface) : {pred_curr:.2f} °C  [{model_used}]")
            print(f"  Model prediction (new surface)   : {pred_new:.2f} °C")
            print()
            print(f"  Temperature delta (model)        : {delta:+.2f} °C")
            print(f"  True baseline temperature        : {current_temp:.2f} °C")
            print(f"  ╔══════════════════════════════════════════════════════╗")
            print(f"  ║  Estimated real-world temperature: {estimated_real_temp:.2f} °C"
                  f"  ({delta:+.2f} °C)  ║")
            print(f"  ╚══════════════════════════════════════════════════════╝")
            if abs(delta) < 0.1:
                print("  (Negligible temperature change for this scenario.)")

            # Extrapolation warning if applicable
            out_lr = (new_green < green_min or new_green > green_max or
                      new_imp   < imp_min   or new_imp   > imp_max)
            if out_lr:
                print("\n  NOTE: Scenario is outside training data range — "
                      "Linear Regression used (extrapolation).")

        except ValueError:
            print("  ERROR: Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting simulator.")
            break


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    df = load_and_merge_data()
    lr_model, X_lr, gb_model, X_tree, processed_df = train_and_evaluate_models(df)
    validate_predictions(gb_model, X_tree, processed_df)
    run_simulator(lr_model, X_lr, gb_model, X_tree, processed_df)


if __name__ == "__main__":
    main()

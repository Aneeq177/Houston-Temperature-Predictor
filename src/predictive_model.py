"""
predictive_model.py
-------------------
Trains machine learning models (Multiple Linear Regression & Random Forest) 
to predict average summer temperature based on surface composition.
Includes cross-validation, feature importance extraction, and an interactive 
"What-If" scenario simulator.

Usage
-----
    python src/predictive_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_CSV  = PROJECT_ROOT / "data" / "weather_stations.csv"
SURFACE_CSV  = PROJECT_ROOT / "data" / "surface_analysis.csv"

def load_and_merge_data() -> pd.DataFrame:
    """Loads and merges the weather and surface data."""
    if not WEATHER_CSV.exists() or not SURFACE_CSV.exists():
        print("Error: Missing data files. Make sure CV pipeline has run.")
        sys.exit(1)
        
    weather = pd.read_csv(WEATHER_CSV)
    surface = pd.read_csv(SURFACE_CSV)
    
    # Merge datasets
    df = weather.merge(surface, on="Station_ID", how="inner")
    df = df.dropna(subset=["Avg_Summer_Temp", "Pct_Green", "Pct_Impervious"])
    
    return df

def train_and_evaluate_models(df: pd.DataFrame):
    """
    Trains and evaluates Linear Regression and Random Forest models using
    5-fold cross-validation. Both models are fit on the full dataset after
    evaluation so they are available for downstream use.

    Returns (lr_model, rf_model, X, df) where:
      lr_model — used by the simulator (LR extrapolates; RF does not)
      rf_model — used for feature importance (RF handles non-linearity better)
    """
    print("=" * 60)
    print("  PHASE 2: PREDICTIVE MODEL TRAINING")
    print("=" * 60)

    X = df[["Pct_Green", "Pct_Impervious", "Latitude", "Longitude"]]
    y = df["Avg_Summer_Temp"]

    # 5-fold CV — single pass per model using cross_validate (avoids fitting
    # the model twice just to get R² and RMSE separately)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scoring = {"r2": "r2", "mse": "neg_mean_squared_error"}

    # ── Model 1: Multiple Linear Regression (baseline) ─────────────────────
    lr_model  = LinearRegression()
    lr_cv     = cross_validate(lr_model, X, y, cv=kf, scoring=cv_scoring)
    lr_r2     = lr_cv["test_r2"]
    lr_rmse   = np.sqrt(-lr_cv["test_mse"])   # neg_MSE → MSE → RMSE

    print("\n--- Model: Multiple Linear Regression ---")
    print(f"  Cross-Validated R²  : {lr_r2.mean():.3f}  (±{lr_r2.std():.3f})")
    print(f"  Cross-Validated RMSE: {lr_rmse.mean():.3f} °C  (±{lr_rmse.std():.3f})")

    # ── Model 2: Random Forest Regressor (non-linear) ──────────────────────
    rf_model  = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf_cv     = cross_validate(rf_model, X, y, cv=kf, scoring=cv_scoring)
    rf_r2     = rf_cv["test_r2"]
    rf_rmse   = np.sqrt(-rf_cv["test_mse"])

    print("\n--- Model: Random Forest Regressor ---")
    print(f"  Cross-Validated R²  : {rf_r2.mean():.3f}  (±{rf_r2.std():.3f})")
    print(f"  Cross-Validated RMSE: {rf_rmse.mean():.3f} °C  (±{rf_rmse.std():.3f})")

    # ── Winner comparison ───────────────────────────────────────────────────
    print("\n--- Model Comparison ---")
    winner = "Linear Regression" if lr_r2.mean() >= rf_r2.mean() else "Random Forest"
    print(f"  Higher CV R²: {winner}")

    # ── Fit both models on ALL data ─────────────────────────────────────────
    # Required so both models are usable for inference and feature inspection.
    lr_model.fit(X, y)
    rf_model.fit(X, y)

    print("\n--- Linear Regression Coefficients ---")
    geo_features = {"Latitude", "Longitude"}
    for feat, coef in zip(X.columns, lr_model.coef_):
        unit = "°C per 1°" if feat in geo_features else "°C per 1%"
        print(f"  {feat:20}: {coef:+.4f} {unit}")
    print(f"  {'Intercept':20}: {lr_model.intercept_:.4f} °C")

    print("\n--- Feature Importance (Random Forest) ---")
    for feat, imp in zip(X.columns, rf_model.feature_importances_):
        print(f"  {feat:20}: {imp:.1%}")

    return lr_model, rf_model, X, df

def run_simulator(lr_model: LinearRegression, X_train: pd.DataFrame, df: pd.DataFrame):
    """
    Interactive 'What-If' microclimate simulator.

    Uses the LINEAR REGRESSION model — not Random Forest — because RF cannot
    extrapolate outside the range of training data.  When a user enters a
    hypothetical scenario (e.g. 60% green space in a currently 20% green
    neighbourhood), the inputs likely fall outside the observed distribution.
    LR extrapolates linearly, which is the correct behaviour for a simulator.
    RF would silently clamp its output to values seen during training, giving
    a misleadingly narrow and inaccurate prediction range.

    Geographic anchor
    -----------------
    Because Latitude and Longitude are model features, every prediction needs
    a location.  The simulator uses the MEAN lat/lon of the training grid —
    a generic "central Houston" location — so that the surface-composition
    inputs alone drive the temperature change, not the coordinates of any
    one specific station.  Using the hottest station's coordinates (the
    previous behaviour) would permanently embed that location's geographic
    heat into every prediction, making it impossible to predict below that
    station's actual temperature regardless of surface inputs.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: 'WHAT-IF' MICROCLIMATE SIMULATOR")
    print("  (Predictions use Linear Regression — valid for extrapolation)")
    print("=" * 60)

    # ── Geographic anchor: mean of the training grid ──────────────────────
    # Keeps location contribution constant across all scenarios so only the
    # surface-composition inputs drive predicted temperature changes.
    anchor_lat = X_train["Latitude"].mean()
    anchor_lon = X_train["Longitude"].mean()

    # ── Training-data bounds for extrapolation warnings ───────────────────
    green_min, green_max = X_train["Pct_Green"].min(),      X_train["Pct_Green"].max()
    imp_min,   imp_max   = X_train["Pct_Impervious"].min(), X_train["Pct_Impervious"].max()

    # ── Reference temperatures ────────────────────────────────────────────
    hottest      = df.loc[df["Avg_Summer_Temp"].idxmax()]
    coolest      = df.loc[df["Avg_Summer_Temp"].idxmin()]
    city_avg_temp = df["Avg_Summer_Temp"].mean()

    # What does the model predict at the city-average surface composition
    # and the anchor location?  This is the "baseline" the simulator moves from.
    city_avg_green = df["Pct_Green"].mean()
    city_avg_imp   = df["Pct_Impervious"].mean()
    baseline_X     = pd.DataFrame({
        "Pct_Green":      [city_avg_green],
        "Pct_Impervious": [city_avg_imp],
        "Latitude":       [anchor_lat],
        "Longitude":      [anchor_lon],
    })
    baseline_pred = lr_model.predict(baseline_X)[0]

    print(f"\n  Geographic anchor : {anchor_lat:.4f}°N, {anchor_lon:.4f}°W  (mean of grid)")
    print(f"\n  Reference temperatures (observed):")
    print(f"    City average — {city_avg_temp:.2f} °C  "
          f"(Green {city_avg_green:.1f}%,  Impervious {city_avg_imp:.1f}%)")
    print(f"    Hottest      — {hottest['Station_ID']}: {hottest['Avg_Summer_Temp']:.2f} °C  "
          f"(Green {hottest['Pct_Green']:.1f}%,  Impervious {hottest['Pct_Impervious']:.1f}%)")
    print(f"    Coolest      — {coolest['Station_ID']}: {coolest['Avg_Summer_Temp']:.2f} °C  "
          f"(Green {coolest['Pct_Green']:.1f}%,  Impervious {coolest['Pct_Impervious']:.1f}%)")
    print(f"\n  Model baseline at city-avg composition : {baseline_pred:.2f} °C")
    print(f"  Training surface ranges:")
    print(f"    Pct_Green      : {green_min:.1f}% – {green_max:.1f}%")
    print(f"    Pct_Impervious : {imp_min:.1f}% – {imp_max:.1f}%")
    print("-" * 60)

    while True:
        try:
            print("\nEnter a hypothetical scenario (blank line to exit):")
            green_input = input("  New % Green Space  [0–100]: ").strip()
            if not green_input:
                break

            imp_input = input("  New % Impervious   [0–100]: ").strip()
            if not imp_input:
                break

            new_green = float(green_input)
            new_imp   = float(imp_input)

            # ── Input validation ──────────────────────────────────────────
            if not (0.0 <= new_green <= 100.0) or not (0.0 <= new_imp <= 100.0):
                print("  ERROR: Percentages must be between 0 and 100. Try again.")
                continue

            if new_green + new_imp > 100.0:
                print(f"  ERROR: Green ({new_green:.1f}%) + Impervious ({new_imp:.1f}%) "
                      f"= {new_green + new_imp:.1f}% exceeds 100%. Try again.")
                continue

            # ── Extrapolation warning ─────────────────────────────────────
            out_of_range = (
                new_green < green_min or new_green > green_max or
                new_imp   < imp_min   or new_imp   > imp_max
            )
            if out_of_range:
                print("  NOTE: Input is outside the training data range — "
                      "prediction is an extrapolation.")

            # ── Predict at anchor location ────────────────────────────────
            # Latitude and Longitude are held at the grid mean so that only
            # the surface composition inputs affect the predicted temperature.
            new_X = pd.DataFrame({
                "Pct_Green":      [new_green],
                "Pct_Impervious": [new_imp],
                "Latitude":       [anchor_lat],
                "Longitude":      [anchor_lon],
            })
            predicted_temp = lr_model.predict(new_X)[0]

            diff_city = predicted_temp - city_avg_temp
            diff_hot  = predicted_temp - hottest['Avg_Summer_Temp']
            diff_cool = predicted_temp - coolest['Avg_Summer_Temp']
            diff_base = predicted_temp - baseline_pred

            print(f"\n  => Predicted Temperature        : {predicted_temp:.2f} °C")
            print(f"  => vs. City average ({city_avg_temp:.2f} °C)  : {diff_city:+.2f} °C")
            print(f"  => vs. Model baseline ({baseline_pred:.2f} °C): {diff_base:+.2f} °C")
            print(f"  => vs. Hottest station           : {diff_hot:+.2f} °C")
            print(f"  => vs. Coolest station           : {diff_cool:+.2f} °C")
            print(f"  => Implied 'Other' surface       : {100 - new_green - new_imp:.1f}%")

        except ValueError:
            print("  ERROR: Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting simulator.")
            break

def main():
    df = load_and_merge_data()
    lr_model, rf_model, X, processed_df = train_and_evaluate_models(df)
    run_simulator(lr_model, X, processed_df)

if __name__ == "__main__":
    main()

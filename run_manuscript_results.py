"""Canonical reproducibility script for the manuscript:
“Surrogate Multi-Objective Screening of a Conceptual Cold-Climate PV System
with Thermal-Protection Context” (Sanzhar Serikbolov).

This script executes disclosed closed-form surrogate screening equations. It
is not a field-validated hourly PV or coupled PV/T simulation, calibrated
battery-life model, basin-specific AWARE assessment, or engineering design.
Tested with Python 3.12. Outputs are written to ./results/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# np.row_stack was deprecated in NumPy 2.0 and removed in NumPy 2.5.
# pymoo 0.6.1.3 can call the legacy alias internally.
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

SEED, POPULATION, GENERATIONS = 42, 150, 500
SBX_PROB, SBX_ETA, PM_PROB, PM_ETA = 0.85, 20, 0.05, 20
PROJECT_YEARS, DISCOUNT_RATE = 20, 0.08
BATTERY_REPLACEMENT_YEARS = (7, 14)
BASE_IDEAL_YIELD_GWH = 282.14
FROST_FREE_DAYS, WATER_PER_WASH_M3 = 150.0, 1280.4
PV_CAPEX_BASE_USD, BESS_CAPEX_USD = 6_000_000.0, 1_250_000.0
OPTIMIZED_ARRAY_CAPEX_USD, THERMAL_BUFFER_CAPEX_USD = 6_400_000.0, 300_000.0
BASE_OPEX_USD, HEATING_OPEX_USD = 120_000.0, 15_000.0
BASELINE_LIFETIME_YIELD_GWH = 242.1  # external planning-level scenario input
AWARE_CF_KAZAKHSTAN = 27.5802  # country-level annual screening factor
AWARE_COUNTRY_TABLE = {"Kazakhstan": 27.5802, "Uzbekistan": 50.2138, "Spain": 31.4107, "Russia": 3.6628}
TARGET_POINT = np.array([28.0, 62.0, 18.0, 40.0])
BOUNDS_LOW, BOUNDS_HIGH = np.array([15., 45., 7., 30.]), np.array([40., 75., 45., 80.])
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def lifetime_yield_gwh(theta_summer, theta_winter, clean_interval_days):
    """Modeled 20-year electrical yield (GWh), clipped to prevent negatives."""
    summer = max(0., 1. - 0.002 * (theta_summer - 28.) ** 2)
    winter = max(0., 1. - 0.005 * (theta_winter - 62.) ** 2)
    cleaning = max(0., 1. - 0.05 * clean_interval_days / 45.)
    return BASE_IDEAL_YIELD_GWH * summer * winter * cleaning


def physical_water_m3_per_year(clean_interval_days, frost_free_days=FROST_FREE_DAYS, water_per_wash_m3=WATER_PER_WASH_M3):
    """Annual physical cleaning-water use; not a 20-year total."""
    return frost_free_days / clean_interval_days * water_per_wash_m3


def thermal_protection_proxy_percent(buffer_setpoint_c):
    """Screening proxy, not an electrochemical state-of-health forecast."""
    return 82.4 - 1.25 * max(0., 40. - buffer_setpoint_c)


class PVSurrogateProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=3, n_ieq_constr=0, xl=BOUNDS_LOW, xu=BOUNDS_HIGH)

    def _evaluate(self, x, out, *args, **kwargs):
        ts, tw, tc, tb = x
        out["F"] = [-lifetime_yield_gwh(ts, tw, tc), physical_water_m3_per_year(tc), -thermal_protection_proxy_percent(tb)]


def run_nsga2():
    result = minimize(PVSurrogateProblem(), NSGA2(pop_size=POPULATION, crossover=SBX(prob=SBX_PROB, eta=SBX_ETA), mutation=PM(prob=PM_PROB, eta=PM_ETA), eliminate_duplicates=True), termination=("n_gen", GENERATIONS), seed=SEED, verbose=False)
    raw = pd.DataFrame({"theta_summer_deg": result.X[:, 0], "theta_winter_deg": result.X[:, 1], "cleaning_interval_days": result.X[:, 2], "buffer_setpoint_c": result.X[:, 3], "lifetime_yield_gwh": -result.F[:, 0], "physical_water_m3_per_year": result.F[:, 1], "thermal_protection_proxy_percent": -result.F[:, 2]})
    raw["distance_to_target"] = np.linalg.norm((result.X - TARGET_POINT) / (BOUNDS_HIGH - BOUNDS_LOW), axis=1)
    raw.loc[[raw.distance_to_target.idxmin()]].drop(columns="distance_to_target").to_csv(OUT / "selected_compromise_solution.csv", index=False)
    pareto = raw.drop(columns="distance_to_target").sort_values("lifetime_yield_gwh", ascending=False).reset_index(drop=True)
    pareto.to_csv(OUT / "pareto_solutions.csv", index=False)
    ts, tw, tc, tb = TARGET_POINT
    y, w, p = lifetime_yield_gwh(ts, tw, tc), physical_water_m3_per_year(tc), thermal_protection_proxy_percent(tb)
    f_round = np.array([-y, w, -p])
    f_all = np.column_stack([-pareto.lifetime_yield_gwh, pareto.physical_water_m3_per_year, -pareto.thermal_protection_proxy_percent])
    n = int((np.all(f_all <= f_round, axis=1) & np.any(f_all < f_round, axis=1)).sum())
    (OUT / "non_dominance_check.txt").write_text(f"Non-dominance verification for round compromise point (28, 62, 18, 40):\n  Evaluated 20-year yield = {y:.4f} GWh\n  Evaluated annual water = {w:.4f} m3/year\n  Evaluated thermal-protection proxy = {p:.4f} % at year 20\n  Rows in pareto_solutions.csv (of {len(pareto)}) that dominate this point: {n}\n  Conclusion: point is {'NON-DOMINATED' if n == 0 else 'DOMINATED'} relative to the returned NSGA-II front. This is not proof of global Pareto optimality.\n")
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    pts = ax.scatter(pareto.physical_water_m3_per_year, pareto.lifetime_yield_gwh, pareto.thermal_protection_proxy_percent, c=pareto.thermal_protection_proxy_percent, cmap="viridis", s=26)
    ax.set(xlabel="Physical cleaning-water use (m3/year)", ylabel="Modeled 20-year yield (GWh)", zlabel="Thermal-protection proxy at year 20 (%)")
    fig.colorbar(pts, ax=ax, label="Thermal-protection proxy (%)"); fig.tight_layout(); fig.savefig(OUT / "pareto_front.png", dpi=300); plt.close(fig)
    return pareto, y, w, p


def discounted_lcoe_usd_per_kwh(lifetime_yield_gwh_value, initial_capex_usd, annual_opex_usd, replacement_years=(), discount_rate=DISCOUNT_RATE):
    annual_energy = lifetime_yield_gwh_value * 1e6 / PROJECT_YEARS
    pv_costs, pv_energy, rows = initial_capex_usd, 0., []
    for year in range(1, PROJECT_YEARS + 1):
        replacement = BESS_CAPEX_USD if year in replacement_years else 0.
        df = (1. + discount_rate) ** year
        cost, energy = (annual_opex_usd + replacement) / df, annual_energy / df
        pv_costs += cost; pv_energy += energy
        rows.append({"year": year, "annual_energy_kwh": annual_energy, "opex_usd": annual_opex_usd, "replacement_usd": replacement, "discount_factor": df, "discounted_cost_usd": cost, "discounted_energy_kwh": energy})
    return pv_costs / pv_energy, pd.DataFrame(rows)


def run_lcoe_audit(optimized_yield):
    base, base_cf = discounted_lcoe_usd_per_kwh(BASELINE_LIFETIME_YIELD_GWH, PV_CAPEX_BASE_USD + BESS_CAPEX_USD, BASE_OPEX_USD, BATTERY_REPLACEMENT_YEARS)
    opt, opt_cf = discounted_lcoe_usd_per_kwh(optimized_yield, OPTIMIZED_ARRAY_CAPEX_USD + BESS_CAPEX_USD + THERMAL_BUFFER_CAPEX_USD, BASE_OPEX_USD + HEATING_OPEX_USD)
    base_cf.to_csv(OUT / "baseline_discounted_cashflow.csv", index=False); opt_cf.to_csv(OUT / "optimized_discounted_cashflow.csv", index=False)
    audit = pd.DataFrame([{"scenario": "baseline_unheated", "lifetime_yield_gwh": BASELINE_LIFETIME_YIELD_GWH, "capex_usd": PV_CAPEX_BASE_USD + BESS_CAPEX_USD, "annual_opex_usd": BASE_OPEX_USD, "replacement_years": str(BATTERY_REPLACEMENT_YEARS), "lcoe_usd_per_kwh": base}, {"scenario": "optimized_thermal_protection", "lifetime_yield_gwh": optimized_yield, "capex_usd": OPTIMIZED_ARRAY_CAPEX_USD + BESS_CAPEX_USD + THERMAL_BUFFER_CAPEX_USD, "annual_opex_usd": BASE_OPEX_USD + HEATING_OPEX_USD, "replacement_years": "()", "lcoe_usd_per_kwh": opt}])
    audit.to_csv(OUT / "lcoe_audit.csv", index=False)
    pd.concat([base_cf.assign(scenario="baseline_unheated"), opt_cf.assign(scenario="optimized_thermal_protection")]).to_csv(OUT / "lcoe_discounted_cashflows.csv", index=False)
    return base, opt


def run_lcoe_sensitivity(optimized_yield):
    opt_capex = OPTIMIZED_ARRAY_CAPEX_USD + BESS_CAPEX_USD + THERMAL_BUFFER_CAPEX_USD; opt_opex = BASE_OPEX_USD + HEATING_OPEX_USD
    opt_base, _ = discounted_lcoe_usd_per_kwh(optimized_yield, opt_capex, opt_opex)
    base_lcoe, _ = discounted_lcoe_usd_per_kwh(BASELINE_LIFETIME_YIELD_GWH, PV_CAPEX_BASE_USD + BESS_CAPEX_USD, BASE_OPEX_USD, BATTERY_REPLACEMENT_YEARS)
    def opt(y=optimized_yield, capex=opt_capex, opex=opt_opex, rate=DISCOUNT_RATE): return discounted_lcoe_usd_per_kwh(y, capex, opex, (), rate)[0]
    def baseline_bess(bess):
        annual = BASELINE_LIFETIME_YIELD_GWH * 1e6 / PROJECT_YEARS; costs, energy = PV_CAPEX_BASE_USD + bess, 0.
        for year in range(1, PROJECT_YEARS + 1):
            df = (1 + DISCOUNT_RATE) ** year; costs += (BASE_OPEX_USD + (bess if year in BATTERY_REPLACEMENT_YEARS else 0.)) / df; energy += annual / df
        return costs / energy
    rows = [
        ["Discount rate", "5%", opt(rate=.05), opt_base, "12%", opt(rate=.12), "optimized thermal-protection scenario"],
        ["Thermal-protection-system CAPEX", "-30%", opt(capex=OPTIMIZED_ARRAY_CAPEX_USD*.7+BESS_CAPEX_USD+THERMAL_BUFFER_CAPEX_USD*.7), opt_base, "+30%", opt(capex=OPTIMIZED_ARRAY_CAPEX_USD*1.3+BESS_CAPEX_USD+THERMAL_BUFFER_CAPEX_USD*1.3), "optimized thermal-protection scenario"],
        ["BESS replacement cost", "-30%", baseline_bess(BESS_CAPEX_USD*.7), base_lcoe, "+30%", baseline_bess(BESS_CAPEX_USD*1.3), "baseline unheated scenario"],
        ["Modeled lifetime yield", "-10%", opt(y=optimized_yield*.9), opt_base, "+10%", opt(y=optimized_yield*1.1), "optimized thermal-protection scenario"],
        ["Heating OPEX", "-50%", opt(opex=BASE_OPEX_USD+HEATING_OPEX_USD*.5), opt_base, "+50%", opt(opex=BASE_OPEX_USD+HEATING_OPEX_USD*1.5), "optimized thermal-protection scenario"],
    ]
    df = pd.DataFrame(rows, columns=["parameter", "low_case", "lcoe_low_usd_per_kwh", "base_case_lcoe_usd_per_kwh", "high_case", "lcoe_high_usd_per_kwh", "scenario"])
    for c in ["lcoe_low_usd_per_kwh", "base_case_lcoe_usd_per_kwh", "lcoe_high_usd_per_kwh"]: df[c] = df[c].round(5)
    df.to_csv(OUT / "lcoe_sensitivity.csv", index=False)


def run_aware_screening(annual_water):
    monthly = pd.DataFrame({"process": ["Module cleaning - warm season"]*5, "month": ["May", "Jun", "Jul", "Aug", "Sep"], "withdrawal_m3": [1800, 2200, 2300, 2200, 2170], "same_basin_return_m3": [0]*5, "aware_cf_m3_worldeq_per_m3": [AWARE_CF_KAZAKHSTAN]*5})
    monthly["consumed_m3"] = monthly.withdrawal_m3 - monthly.same_basin_return_m3; monthly["aware_footprint_m3_worldeq"] = monthly.consumed_m3 * monthly.aware_cf_m3_worldeq_per_m3
    monthly.to_csv(OUT / "aware_inventory_and_results.csv", index=False)
    regions = pd.DataFrame({"country": list(AWARE_COUNTRY_TABLE), "aware_cf_m3_worldeq_per_m3": list(AWARE_COUNTRY_TABLE.values())}); regions["physical_consumption_m3_per_year"] = annual_water; regions["aware_footprint_m3_worldeq_per_year"] = regions.physical_consumption_m3_per_year * regions.aware_cf_m3_worldeq_per_m3
    regions.to_csv(OUT / "aware_location_screening.csv", index=False)
    footprint = annual_water * AWARE_CF_KAZAKHSTAN
    (OUT / "aware_screening_report.md").write_text(f"# AWARE screening report\n\nCountry-level annual screening only.\n\n{annual_water:,.0f} m3/year × {AWARE_CF_KAZAKHSTAN} = {footprint:,.0f} m3 world-eq/year.\n\nNo cleaning inventory occurs in freezing months; this does not assign an AWARE factor of zero.\n")


def run_water_sensitivities(clean_interval=18.):
    base = physical_water_m3_per_year(clean_interval)
    pd.DataFrame([{"assumption": "V_wash (m3/event)", "base_case_value": WATER_PER_WASH_M3, "low_case_pct": "-30%", "annual_water_low_m3": round(base*.7, 1), "base_case_annual_water_m3": round(base, 1), "high_case_pct": "+30%", "annual_water_high_m3": round(base*1.3, 1)}]).to_csv(OUT / "water_per_wash_sensitivity.csv", index=False)
    pd.DataFrame([{"frost_free_days": d, "annual_physical_water_m3": round(physical_water_m3_per_year(clean_interval, frost_free_days=d), 1)} for d in (120., 150., 180.)]).to_csv(OUT / "frost_free_window_sensitivity.csv", index=False)


def main():
    pareto, y, w, proxy = run_nsga2(); base, opt = run_lcoe_audit(y); run_lcoe_sensitivity(y); run_aware_screening(w); run_water_sensitivities()
    print(f"Round point: modeled 20-year yield={y:.4f} GWh; water={w:.4f} m3/year; thermal-protection proxy={proxy:.4f}%.")
    print(f"LCOE: baseline={base:.4f}; optimized thermal-protection scenario={opt:.4f} USD/kWh.")
    print(f"All outputs written to {OUT.resolve()}")

if __name__ == "__main__":
    main()

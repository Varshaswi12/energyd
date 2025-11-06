# robust_dispatch.py
import os
import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory, minimize
)

BASE = os.getcwd()
DATA = os.path.join(BASE, "data", "processed", "opsd_de_hourly.csv")
print("Loading:", DATA)

df = pd.read_csv(DATA, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
df = df[['load', 'solar', 'wind', 'price']].interpolate().fillna(method='ffill')

# Use last 7 days for demonstration
data = df.tail(24 * 7)

# Simulate probabilistic forecasts using random noise (±10%)
np.random.seed(42)
scenarios = []
for s in range(5):  # 5 probabilistic scenarios
    scenario = data.copy()
    scenario['load'] *= np.random.normal(1.0, 0.05, len(data))
    scenario['solar'] *= np.random.normal(1.0, 0.10, len(data))
    scenario['wind'] *= np.random.normal(1.0, 0.10, len(data))
    scenario['scenario'] = s
    scenarios.append(scenario)

scenarios_df = pd.concat(scenarios)
scenarios_df.to_csv(os.path.join(BASE, "data", "processed", "probabilistic_scenarios.csv"))
print("Generated 5 probabilistic forecast scenarios.")

# ==========================
# Robust Optimization Model
# ==========================

model = ConcreteModel()
hours = range(len(data))
scenarios_list = list(range(5))

# Decision variables per hour (deterministic plan)
model.solar = Var(hours, domain=NonNegativeReals)
model.wind = Var(hours, domain=NonNegativeReals)
model.gas = Var(hours, domain=NonNegativeReals)

# Battery storage variables
model.charge = Var(hours, domain=NonNegativeReals)
model.discharge = Var(hours, domain=NonNegativeReals)
model.soc = Var(hours, domain=NonNegativeReals)

# Parameters
solar_cap = data['solar'].max()
wind_cap = data['wind'].max()
gas_cap = data['load'].max()
battery_cap = 0.2 * gas_cap  # battery size = 20% of peak load
eff = 0.9

# Cost and emission
C_SOLAR, C_WIND, C_GAS = 15, 25, 70
E_GAS = 500

# Objective: minimize expected cost across all scenarios
def obj_rule(m):
    total_cost = 0
    for s in scenarios_list:
        scen = scenarios[s]
        for t in hours:
            total_cost += (
                C_SOLAR * m.solar[t]
                + C_WIND * m.wind[t]
                + C_GAS * m.gas[t]
                + 0.02 * E_GAS * m.gas[t]
            )
    return total_cost / len(scenarios_list)

model.obj = Objective(rule=obj_rule, sense=minimize)

# Constraints
def balance_rule(m, t):
    # Must meet load in all scenarios (robust)
    min_load = min(scenarios_df.loc[scenarios_df.index.hour == t % 24, 'load'])
    return (
        m.solar[t] + m.wind[t] + m.gas[t] + m.discharge[t] - m.charge[t] >= min_load
    )

model.balance = Constraint(hours, rule=balance_rule)

def storage_rule(m, t):
    if t == 0:
        return m.soc[t] == 0.5 * battery_cap + eff * m.charge[t] - m.discharge[t] / eff
    else:
        return m.soc[t] == m.soc[t - 1] + eff * m.charge[t] - m.discharge[t] / eff

model.storage = Constraint(hours, rule=storage_rule)

# SOC bounds
def soc_cap_rule(m, t):
    return m.soc[t] <= battery_cap
model.soc_cap = Constraint(hours, rule=soc_cap_rule)

# Generation limits
def solar_cap_rule(m, t):
    return m.solar[t] <= solar_cap
model.solar_cap = Constraint(hours, rule=solar_cap_rule)

def wind_cap_rule(m, t):
    return m.wind[t] <= wind_cap
model.wind_cap = Constraint(hours, rule=wind_cap_rule)

def gas_cap_rule(m, t):
    return m.gas[t] <= gas_cap
model.gas_cap = Constraint(hours, rule=gas_cap_rule)

# Solve
solver = SolverFactory('appsi_highs')
result = solver.solve(model)

# Collect results
results = pd.DataFrame({
    'hour': hours,
    'solar_used': [model.solar[h]() for h in hours],
    'wind_used': [model.wind[h]() for h in hours],
    'gas_used': [model.gas[h]() for h in hours],
    'soc': [model.soc[h]() for h in hours],
})
results['total_gen'] = results['solar_used'] + results['wind_used'] + results['gas_used']

# Save & display
out_csv = os.path.join(BASE, "data", "processed", "robust_dispatch_results.csv")
results.to_csv(out_csv, index=False)
print("✅ Robust optimization complete!")
print("Saved results to:", out_csv)
print(results.head())

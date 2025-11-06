# optimize_dayahead.py
import os
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, SolverFactory, minimize

BASE = os.getcwd()
DATA = os.path.join(BASE, "data", "processed", "opsd_de_hourly.csv")
print("Loading:", DATA)
df = pd.read_csv(DATA, parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Select columns (some may be NaN, handle safely)
df = df[['load', 'solar', 'wind', 'price']].interpolate().fillna(method='ffill')

# Aggregate by day (sum load, mean price)
daily = df.resample('D').agg({'load': 'sum', 'solar': 'sum', 'wind': 'sum', 'price': 'mean'}).dropna()

# Take last 30 days for optimization example
data = daily.tail(30)
days = list(range(len(data)))

# Create Pyomo model
model = ConcreteModel()

# Decision variables: energy supplied by solar, wind, gas (MW)
model.solar = Var(days, domain=NonNegativeReals)
model.wind = Var(days, domain=NonNegativeReals)
model.gas = Var(days, domain=NonNegativeReals)

# Parameters
solar_cap = data['solar'].max()
wind_cap = data['wind'].max()
gas_cap = data['load'].max()  # gas can cover any residual load

# Costs ($/MWh)
COST_SOLAR = 15
COST_WIND = 25
COST_GAS = 70

# Emissions (kg CO₂/MWh)
EM_SOLAR = 0
EM_WIND = 0
EM_GAS = 500

# Objective: minimize total cost + emissions penalty
def obj_rule(m):
    cost = sum(COST_SOLAR*m.solar[d] + COST_WIND*m.wind[d] + COST_GAS*m.gas[d] for d in days)
    emissions = sum(EM_SOLAR*m.solar[d] + EM_WIND*m.wind[d] + EM_GAS*m.gas[d] for d in days)
    return cost + 0.02 * emissions  # small weight for emissions
model.obj = Objective(rule=obj_rule, sense=minimize)

# Constraints
def demand_balance(m, d):
    return m.solar[d] + m.wind[d] + m.gas[d] >= data['load'].iloc[d]
model.demand_constraint = Constraint(days, rule=demand_balance)

def solar_cap_rule(m, d):
    return m.solar[d] <= solar_cap
model.solar_cap = Constraint(days, rule=solar_cap_rule)

def wind_cap_rule(m, d):
    return m.wind[d] <= wind_cap
model.wind_cap = Constraint(days, rule=wind_cap_rule)

def gas_cap_rule(m, d):
    return m.gas[d] <= gas_cap
model.gas_cap = Constraint(days, rule=gas_cap_rule)

# Solve
solver = SolverFactory('appsi_highs')
result = solver.solve(model, tee=False)

# Extract results
results = pd.DataFrame({
    'day': data.index,
    'solar_used': [model.solar[d]() for d in days],
    'wind_used': [model.wind[d]() for d in days],
    'gas_used': [model.gas[d]() for d in days],
    'demand': data['load'].values
})

results['total_cost'] = (
    results['solar_used']*COST_SOLAR +
    results['wind_used']*COST_WIND +
    results['gas_used']*COST_GAS
)

print("\n✅ Optimization done!")
print(results.head())

# Save results
out_file = os.path.join(BASE, "data", "processed", "dayahead_optimization.csv")
results.to_csv(out_file, index=False)
print("Saved optimized schedule:", out_file)

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# Constants
constants = {
    "time_periods": 24,
    "energy_sources": ["Grid", "Solar", "Battery"],
    "scenarios": ["S_high", "S_avg", "S_low"],
    "probs": {"S_high": 0.3, "S_avg": 0.4, "S_low": 0.3}
}

# Input Data (manually created or loaded from a file)

data = {
    'Cost_grid': {'S_high': 30, 'S_avg': 70, 'S_low': 120},  # Grid cost for each scenario
    'Cost_solar': 0,  # Solar has zero marginal cost
    'Cost_battery': 10,
    'Export_income': {'S_high': 100, 'S_avg': 50, 'S_low': 20},  # Export income for each scenario
    'aFRR_price': 220,  # aFRR price per scenario
    'Energy_demand':[3500] * constants['time_periods'], #Constant demand for simplicity
   
    'Solar_generation': {  # Solar generation also varies
        'S_high': [200] * constants['time_periods'],  # High solar generation
        'S_avg': [80] * constants['time_periods'],    # Average solar generation
        'S_low': [20] * constants['time_periods'],    # Low solar generation
    },
    'Battery_charge_eff': 0.95,
    'Battery_discharge_eff': 0.95,
    'Max_battery_capacity': 100,
    'Max_battery_charge': 100,
    'Max_battery_discharge': 0,
    'Grid_capacity': 100,
    'Initial_battery_storage': 30
}


# Define the stochastic model setup
def StochasticModelSetUp(data, constants):
    # Create a concrete model
    m = pyo.ConcreteModel()

    # Sets
    m.T = pyo.RangeSet(1, constants["time_periods"])  # Time periods
    m.I = pyo.Set(initialize=constants["energy_sources"])  # Energy sources
    m.S = pyo.Set(initialize=constants["scenarios"])  # Scenarios

    # Parameters
    m.C = pyo.Param(m.S, m.I, initialize={(s, i): data['Cost_grid'][s] if i != 'Solar' else data['Cost_solar'] for s in constants['scenarios'] for i in constants['energy_sources']})
    m.C_exp = pyo.Param(m.S, initialize={s: data['Export_income'][s] for s in constants['scenarios']})
    m.P_aFRR = pyo.Param(initialize=data['aFRR_price'])
    m.D = pyo.Param(m.T, initialize={t: data['Energy_demand'][t-1] for t in range(1, constants["time_periods"] + 1)})
    m.G_solar = pyo.Param(m.T, m.S, initialize={(t, s): data['Solar_generation'][s][t-1] for t in range(1, constants["time_periods"] + 1) for s in constants["scenarios"]})
    m.eta_charge = pyo.Param(initialize=data['Battery_charge_eff'])
    m.eta_discharge = pyo.Param(initialize=data['Battery_discharge_eff'])
    m.E_max = pyo.Param(initialize=data['Max_battery_capacity'])
    m.P_charge_max = pyo.Param(initialize=data['Max_battery_charge'])
    m.P_discharge_max = pyo.Param(initialize=data['Max_battery_discharge'])
    m.G_max = pyo.Param(initialize=data['Grid_capacity'])
    m.I_INIT = pyo.Param(initialize=data['Initial_battery_storage'])
    m.pi = pyo.Param(m.S, initialize=constants["probs"])  # Probability of each scenario

    # Variables
    m.x_aFRR = pyo.Var(m.T, within=pyo.NonNegativeReals)  # aFRR reserve (assumed to be scenario-independent)
    m.y_supply = pyo.Var(m.T, m.S, m.I, within=pyo.NonNegativeReals)  # Energy supply from sources
    m.z_export = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Energy exported to the grid
    m.q_charge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery charge
    m.q_discharge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery discharge
    m.e_storage = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Battery energy storage (shared across scenarios)

    # Objective Function: Expected cost over all scenarios
    def Obj(m):
        return -sum(m.x_aFRR[t] * m.P_aFRR for t in m.T) + \
               sum(m.pi[s] * sum(sum(m.y_supply[t, s, i] * m.C[s, i] - m.z_export[t, s] * m.C_exp[s] for i in m.I) for t in m.T) for s in m.S)
    m.obj = pyo.Objective(rule=Obj, sense=pyo.minimize)

    # Constraints
    def EnergyBalance(m, t, s):
        return m.D[t] - m.G_solar[t, s] + m.x_aFRR[t] == sum(m.y_supply[t, s, i] for i in m.I) - m.z_export[t, s] + \
               m.q_discharge[t, s] - m.eta_charge * m.q_charge[t, s]
    m.EnergyBalance = pyo.Constraint(m.T, m.S, rule=EnergyBalance)

    def StorageDynamics(m, t, s):
        if t == 1:
            return m.e_storage[t] == m.I_INIT + m.q_charge[t, s] - m.q_discharge[t, s] / m.eta_discharge
        else:
            return m.e_storage[t] == m.e_storage[t-1] + m.q_charge[t, s] - m.q_discharge[t, s] / m.eta_discharge
    m.StorageDynamics = pyo.Constraint(m.T, m.S, rule=StorageDynamics)

    def BatteryLimits(m, t):
        return m.e_storage[t] <= m.E_max
    m.BatteryLimits = pyo.Constraint(m.T, rule=BatteryLimits)

    def ChargeLimit(m, t, s):
        return m.q_charge[t, s] <= m.P_charge_max
    m.ChargeLimit = pyo.Constraint(m.T, m.S, rule=ChargeLimit)

    def DischargeLimit(m, t, s):
        return m.q_discharge[t, s] <= m.P_discharge_max
    m.DischargeLimit = pyo.Constraint(m.T, m.S, rule=DischargeLimit)

    def BatterySupplyLimit(m, t, s):
        return m.y_supply[t, s, 'Battery'] <= m.eta_discharge * m.e_storage[t]
    m.BatterySupplyLimit = pyo.Constraint(m.T, m.S, rule=BatterySupplyLimit)

    def ImportLimit(m, t, s):
        return m.y_supply[t, s, 'Grid'] <= m.G_max
    m.ImportLimit = pyo.Constraint(m.T, m.S, rule=ImportLimit)

    def ExportLimit(m, t, s):
        return m.z_export[t, s] + m.x_aFRR[t] <= m.G_max
    m.ExportLimit = pyo.Constraint(m.T, m.S, rule=ExportLimit)

    def SolarPowerLimit(m, t, s):
        return m.y_supply[t, s, 'Solar'] <= m.G_solar[t, s]
    m.SolarPowerLimit = pyo.Constraint(m.T, m.S, rule=SolarPowerLimit)

    return m

# Solve the model
def SolveModel(m):
    solver = SolverFactory('gurobi')
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = solver.solve(m, tee=True)
    return results, m

# Display results
def DisplayResults(m):
    return print(m.display(), m.dual.display())

m = StochasticModelSetUp(data, constants)
SolveModel(m)
DisplayResults(m)

def SaveResultsToFile(m, file_name):
    results = {
        'time_period': [],
        'scenario': [],
        'grid_supply': [],
        'solar_supply': [],
        'battery_supply': [],
        'export': [],
        'battery_charge': [],
        'battery_discharge': [],
        'battery_storage': []
    }
    
    for t in m.T:
        for s in m.S:
            results['time_period'].append(t)
            results['scenario'].append(s)
            results['grid_supply'].append(pyo.value(m.y_supply[t, s, 'Grid']))
            results['solar_supply'].append(pyo.value(m.y_supply[t, s, 'Solar']))
            results['battery_supply'].append(pyo.value(m.y_supply[t, s, 'Battery']))
            results['export'].append(pyo.value(m.z_export[t, s]))
            results['battery_charge'].append(pyo.value(m.q_charge[t, s]))
            results['battery_discharge'].append(pyo.value(m.q_discharge[t, s]))
            results['battery_storage'].append(pyo.value(m.e_storage[t]))
    
    df = pd.DataFrame(results)
    df.to_excel(file_name, index=False)
    return file_name


output_file = 'stochastic_model_output.xlsx'
SaveResultsToFile(m, output_file)
print(f'Results saved to: {output_file}')

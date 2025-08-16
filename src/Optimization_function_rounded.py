# In the name of God

import pyomo.environ as pyo
import numpy as np
from sympy.sets.fancysets import Reals


def round_value(value, digits=2):
    return round(value, digits) if abs(value) > 1e-2 else 0

# Creating Energy Management System (EMS) optimization block model:
def create_ems_opti_model(imp_coef, cons_pred, wind_pred, pv_pred, batt_init_SoC, hydr_init_SoC):
    # -------------------------------- Building model ----------------------------------------
    model = pyo.ConcreteModel(name='(EMS-Opti)')
    # -------------------------------- Optimization window ----------------------------------------
    window = len(wind_pred)
    model.T = pyo.Set(initialize=list(range(window)))
    model.T_1 = pyo.Set(initialize=list(range(window - 1)))
    # -------------------------------- Rounding input values --------------------------------------
    cons_pred = {t: round_value(val) for t, val in cons_pred.items()}
    wind_pred = {t: round_value(val) for t, val in wind_pred.items()}
    pv_pred = {t: round_value(val) for t, val in pv_pred.items()}
    # -------------------------------- Imported parameters ----------------------------------------
    # wind production at each time step:
    model.p_wind = pyo.Param(model.T, initialize=wind_pred)
    # pv production at each time step:
    model.p_pv = pyo.Param(model.T, initialize=pv_pred)
    # loads consumption at each time step:
    model.p_consumption = pyo.Param(model.T, initialize=cons_pred)
    # importance coefficients of each part of objective function
    obj_num = len(imp_coef)
    model.N = pyo.Set(initialize=list(range(obj_num)))
    model.importance_coefficients = pyo.Param(model.N, initialize=imp_coef)
    # -------------------------------- Configured parameters ----------------------------------------
    # -------------------------------- BESS ----------------------------------------
    # maximum capacity of battery:
    model.battery_capacity = pyo.Param(initialize=round_value(500.))
    # Initial state of charge of battery:
    # model.battery_initial_SoC = pyo.Param(initialize=batt_init_SoC)
    # maximum charging rate of battery:
    model.battery_max_charging_rate = pyo.Param(initialize=round_value(400.))
    # maximum discharging rate of battery:
    model.battery_max_discharging_rate = pyo.Param(initialize=round_value(400.))
    # charging efficiency of battery:
    model.battery_charging_efficiency = pyo.Param(initialize=round_value(0.98))
    # discharging efficiency of battery:
    model.battery_discharging_efficiency = pyo.Param(initialize=round_value(0.98))
    # minimum SoC of battery:
    model.battery_min_SoC = pyo.Param(initialize=round_value(0.))
    # -------------------------------- HESS ----------------------------------------
    # maximum capacity of hydrogen:
    model.hydrogen_capacity = pyo.Param(initialize=round_value(1670.))
    # Initial state of charge of hydrogen:
    # model.hydrogen_initial_SoC = pyo.Param(initialize=hydr_init_SoC)
    # maximum charging rate of hydrogen:
    model.hydrogen_max_charging_rate = pyo.Param(initialize=round_value(55.))
    # maximum discharging rate of hydrogen:
    model.hydrogen_max_discharging_rate = pyo.Param(initialize=round_value(100.))
    # charging efficiency of hydrogen:
    model.hydrogen_charging_efficiency = pyo.Param(initialize=round_value(0.64))
    # discharging efficiency of hydrogen:
    model.hydrogen_discharging_efficiency = pyo.Param(initialize=round_value(0.50))
    # minimum SoC of hydrogen:
    model.hydrogen_min_SoC = pyo.Param(initialize=round_value(0.))
    # -------------------------------- Diesel generator ----------------------------------------
    # maximum power capacity of diesel generator:
    model.diesel_max_power = pyo.Param(initialize=round_value(90.))
    # -------------------------------- Variables ----------------------------------------
    # -------------------------------- BESS ----------------------------------------
    # charging power of battery:
    model.p_battery_charging = pyo.Var(model.T, bounds=(0, model.battery_max_charging_rate),
                                       domain=pyo.NonNegativeReals)
    # discharging power of battery:
    model.p_battery_discharging = pyo.Var(model.T, bounds=(0, model.battery_max_discharging_rate),
                                          domain=pyo.NonNegativeReals)
    # state of battery charging operation (1 for charging 0 for not charging):
    model.battery_charging_operation_state = pyo.Var(model.T, domain=pyo.Binary)
    # state of battery discharging operation (1 for discharging 0 for not discharging):
    model.battery_discharging_operation_state = pyo.Var(model.T, domain=pyo.Binary)
    # linear variable for charging power of battery:
    model.p_battery_charging_linear = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # linear variable for discharging power of battery:
    model.p_battery_discharging_linear = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # state of charge for battery:
    model.battery_SoC = pyo.Var(model.T, bounds=(model.battery_min_SoC, model.battery_capacity),
                                domain=pyo.NonNegativeReals)
    # model.battery_SoC[0] = batt_init_SoC
    # -------------------------------- HESS ----------------------------------------
    # charging power of hydrogen:
    model.p_hydrogen_charging = pyo.Var(model.T, bounds=(0, model.hydrogen_max_charging_rate),
                                        domain=pyo.NonNegativeReals)
    # discharging power of hydrogen:
    model.p_hydrogen_discharging = pyo.Var(model.T, bounds=(0, model.hydrogen_max_discharging_rate),
                                           domain=pyo.NonNegativeReals)
    # state of hydrogen charging operation (1 for charging 0 for not charging):
    model.hydrogen_charging_operation_state = pyo.Var(model.T, domain=pyo.Binary)
    # state of hydrogen discharging operation (1 for discharging 0 for not discharging):
    model.hydrogen_discharging_operation_state = pyo.Var(model.T, domain=pyo.Binary)
    # linear variable for charging power of hydrogen:
    model.p_hydrogen_charging_linear = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # linear variable for discharging power of hydrogen:
    model.p_hydrogen_discharging_linear = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # state of charge for hydrogen:
    model.hydrogen_SoC = pyo.Var(model.T, bounds=(model.hydrogen_min_SoC, model.hydrogen_capacity),
                                 domain=pyo.NonNegativeReals)
    # model.hydrogen_SoC[0] = hydr_init_SoC
    # -------------------------------- Wind ----------------------------------------
    # curtailment power of wind:
    model.p_wind_curtailment = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # -------------------------------- PV ----------------------------------------
    # curtailment power of pv:
    model.p_pv_curtailment = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    # -------------------------------- Diesel generator ----------------------------------------
    # needed power of diesel
    model.p_diesel = pyo.Var(model.T, bounds=(0, model.diesel_max_power), domain=pyo.NonNegativeReals)

    # -------------------------------- Constraints ----------------------------------------
    # -------------------------------- Battery recursive SoC equation ----------------------------------------
    # Recursive SoC equation of battery:
    def constr_battery_recursive_soc_rule(mdl, t):
        return (mdl.battery_SoC[t + 1] ==
                mdl.battery_SoC[t] +
                mdl.battery_charging_efficiency * mdl.p_battery_charging_linear[t] -
                mdl.p_battery_discharging_linear[t] / mdl.battery_discharging_efficiency)

    model.constr_battery_recursive_soc = pyo.Constraint(model.T_1, rule=constr_battery_recursive_soc_rule)
    # -------------------------------- Battery initial SoC equation ----------------------------------------
    model.constr_battery_soc_0 = pyo.Constraint(expr=model.battery_SoC[0] == batt_init_SoC)

    # -------------------------------- Hydrogen recursive SoC equation ----------------------------------------
    # Recursive SoC equation of hydrogen:
    def constr_hydrogen_recursive_soc_rule(mdl, t):
        return (mdl.hydrogen_SoC[t + 1] ==
                mdl.hydrogen_SoC[t] +
                mdl.hydrogen_charging_efficiency * mdl.p_hydrogen_charging_linear[t] -
                mdl.p_hydrogen_discharging_linear[t] / mdl.hydrogen_discharging_efficiency)

    model.constr_hydrogen_recursive_soc = pyo.Constraint(model.T_1, rule=constr_hydrogen_recursive_soc_rule)
    # -------------------------------- Hydrogen initial SoC equation ----------------------------------------
    model.constr_hydrogen_soc_0 = pyo.Constraint(expr=model.hydrogen_SoC[0] == hydr_init_SoC)

    # -------------------------------- Battery charge/discharge state check ----------------------------------------
    # battery can't be charge and discharge at the same time:
    def constr_battery_charge_discharge_state_check_rule(mdl, t):
        return (mdl.battery_charging_operation_state[t] +
                mdl.battery_discharging_operation_state[t] <= 1)

    model.constr_battery_charge_discharge_state_check = \
        pyo.Constraint(model.T, rule=constr_battery_charge_discharge_state_check_rule)

    # -------------------------------- Hydrogen charge/discharge state check ----------------------------------------
    # hydrogen can't be charge and discharge at the same time:
    def constr_hydrogen_charge_discharge_state_check_rule(mdl, t):
        return (mdl.hydrogen_charging_operation_state[t] +
                mdl.hydrogen_discharging_operation_state[t] <= 1)

    model.constr_hydrogen_charge_discharge_state_check = \
        pyo.Constraint(model.T, rule=constr_hydrogen_charge_discharge_state_check_rule)

    # -------------------------------- Power equality ----------------------------------------
    # In each time step 't' sum of generation and sum of consumption must be equal
    def constr_power_equality_rule(mdl, t):
        return (mdl.p_battery_discharging_linear[t] +
                mdl.p_hydrogen_discharging_linear[t] +
                mdl.p_diesel[t] +
                mdl.p_wind[t] +
                mdl.p_pv[t]
                ==
                mdl.p_battery_charging_linear[t] +
                mdl.p_hydrogen_charging_linear[t] +
                mdl.p_wind_curtailment[t] +
                mdl.p_pv_curtailment[t] +
                mdl.p_consumption[t])

    model.constr_power_equality = pyo.Constraint(model.T, rule=constr_power_equality_rule)

    # -------------------------------- Wind curtailment ----------------------------------------
    # At each time step [t] curtailment power of wind generation must be less than or equal to wind generation:
    def constr_wind_curtailment_rule(mdl, t):
        return mdl.p_wind_curtailment[t] <= mdl.p_wind[t]

    model.constr_wind_curtailment = pyo.Constraint(model.T, rule=constr_wind_curtailment_rule)

    # -------------------------------- PV curtailment ----------------------------------------
    # At each time step [t] curtailment power of pv generation must be less than or equal to pv generation:
    def constr_pv_curtailment_rule(mdl, t):
        return mdl.p_pv_curtailment[t] <= mdl.p_pv[t]

    model.constr_pv_curtailment = pyo.Constraint(model.T, rule=constr_pv_curtailment_rule)

    # -------------------------------- Linearization constraints ----------------------------------------
    # -------------------------------- battery charging Linearization constraints ----------------------------------------
    def constr_linear_battery_charging_1_rule(mdl, t):
        return mdl.p_battery_charging_linear[t] <= mdl.p_battery_charging[t]

    model.constr_linear_battery_charging_1 = pyo.Constraint(model.T, rule=constr_linear_battery_charging_1_rule)

    def constr_linear_battery_charging_2_rule(mdl, t):
        return mdl.p_battery_charging_linear[t] <= mdl.battery_max_charging_rate * mdl.battery_charging_operation_state[
            t]

    model.constr_linear_battery_charging_2 = pyo.Constraint(model.T, rule=constr_linear_battery_charging_2_rule)

    def constr_linear_battery_charging_3_rule(mdl, t):
        return mdl.p_battery_charging_linear[t] >= \
            mdl.p_battery_charging[t] + mdl.battery_max_charging_rate * (mdl.battery_charging_operation_state[t] - 1)

    model.constr_linear_battery_charging_3 = pyo.Constraint(model.T, rule=constr_linear_battery_charging_3_rule)

    # -------------------------------- battery discharging Linearization constraints ----------------------------------------
    def constr_linear_battery_discharging_1_rule(mdl, t):
        return mdl.p_battery_discharging_linear[t] <= mdl.p_battery_discharging[t]

    model.constr_linear_battery_discharging_1 = pyo.Constraint(model.T, rule=constr_linear_battery_discharging_1_rule)

    def constr_linear_battery_discharging_2_rule(mdl, t):
        return mdl.p_battery_discharging_linear[t] <= \
            mdl.battery_max_discharging_rate * mdl.battery_discharging_operation_state[t]

    model.constr_linear_battery_discharging_2 = pyo.Constraint(model.T, rule=constr_linear_battery_discharging_2_rule)

    def constr_linear_battery_discharging_3_rule(mdl, t):
        return mdl.p_battery_discharging_linear[t] >= \
            mdl.p_battery_discharging[t] + mdl.battery_max_discharging_rate * (
                        mdl.battery_discharging_operation_state[t] - 1)

    model.constr_linear_battery_discharging_3 = pyo.Constraint(model.T, rule=constr_linear_battery_discharging_3_rule)

    # -------------------------------- Hydrogen charging Linearization constraints ----------------------------------------
    def constr_linear_hydrogen_charging_1_rule(mdl, t):
        return mdl.p_hydrogen_charging_linear[t] <= mdl.p_hydrogen_charging[t]

    model.constr_linear_hydrogen_charging_1 = pyo.Constraint(model.T, rule=constr_linear_hydrogen_charging_1_rule)

    def constr_linear_hydrogen_charging_2_rule(mdl, t):
        return mdl.p_hydrogen_charging_linear[t] <= mdl.hydrogen_max_charging_rate * \
            mdl.hydrogen_charging_operation_state[t]

    model.constr_linear_hydrogen_charging_2 = pyo.Constraint(model.T, rule=constr_linear_hydrogen_charging_2_rule)

    def constr_linear_hydrogen_charging_3_rule(mdl, t):
        return mdl.p_hydrogen_charging_linear[t] >= \
            mdl.p_hydrogen_charging[t] + mdl.hydrogen_max_charging_rate * (mdl.hydrogen_charging_operation_state[t] - 1)

    model.constr_linear_hydrogen_charging_3 = pyo.Constraint(model.T, rule=constr_linear_hydrogen_charging_3_rule)

    # -------------------------------- Hydrogen discharging Linearization constraints ----------------------------------------
    def constr_linear_hydrogen_discharging_1_rule(mdl, t):
        return mdl.p_hydrogen_discharging_linear[t] <= mdl.p_hydrogen_discharging[t]

    model.constr_linear_hydrogen_discharging_1 = pyo.Constraint(model.T, rule=constr_linear_hydrogen_discharging_1_rule)

    def constr_linear_hydrogen_discharging_2_rule(mdl, t):
        return mdl.p_hydrogen_discharging_linear[t] <= \
            mdl.hydrogen_max_discharging_rate * mdl.hydrogen_discharging_operation_state[t]

    model.constr_linear_hydrogen_discharging_2 = pyo.Constraint(model.T, rule=constr_linear_hydrogen_discharging_2_rule)

    def constr_linear_hydrogen_discharging_3_rule(mdl, t):
        return mdl.p_hydrogen_discharging_linear[t] >= \
            mdl.p_hydrogen_discharging[t] + \
            mdl.hydrogen_max_discharging_rate * (mdl.hydrogen_discharging_operation_state[t] - 1)

    model.constr_linear_hydrogen_discharging_3 = pyo.Constraint(model.T, rule=constr_linear_hydrogen_discharging_3_rule)

    # -------------------------------- Objective Function ----------------------------------------

    def objective_function(mdl):
        # diesel generation over window period have to be minimized:
        f_diesel = sum([mdl.p_diesel[t] for t in mdl.T])
        # curtailed power of wind generation over window period have to be minimized:
        f_wind_curtailed = sum([mdl.p_wind_curtailment[t] for t in mdl.T])
        # curtailed power of pv generation over window period have to be minimum:
        f_pv_curtailed = sum([mdl.p_pv_curtailment[t] for t in mdl.T])
        # charge and discharge loss of battery have to be minimum:
        f_battery_loss = (((1 - mdl.battery_discharging_efficiency) / mdl.battery_discharging_efficiency) *
                          sum([mdl.p_battery_discharging_linear[t] for t in mdl.T]) +
                          (1 - mdl.battery_charging_efficiency) *
                          sum([mdl.p_battery_charging_linear[t] for t in mdl.T]))
        # charge and discharge loss of hydrogen have to be minimum:
        f_hydrogen_loss = (((1 - mdl.hydrogen_discharging_efficiency) / mdl.hydrogen_discharging_efficiency) *
                           sum([mdl.p_hydrogen_discharging_linear[t] for t in mdl.T]) +
                           (1 - mdl.hydrogen_charging_efficiency) *
                           sum([mdl.p_hydrogen_charging_linear[t] for t in mdl.T]))

        return (mdl.importance_coefficients[0] * f_diesel +
                mdl.importance_coefficients[1] * f_wind_curtailed +
                mdl.importance_coefficients[2] * f_pv_curtailed +
                mdl.importance_coefficients[3] * f_battery_loss +
                mdl.importance_coefficients[4] * f_hydrogen_loss)

    model.objective = pyo.Objective(rule=objective_function)

    return model

# Transform a list of timeseries format into a dictionary
def timeseries_to_dictionary(ts):
    ts_flat_list = []
    for sublist in ts.values().tolist():
        for item in sublist:
            ts_flat_list.append(item)

    return dict(enumerate(ts_flat_list))


def create_time_window_dicts(time_series, window_size=24, stride=1):
    """
    Converts a time series into a list of dictionaries using time windows.

    :param time_series: A list of time series values
    :param window_size: The size of the time window (number of hours in each dictionary)
    :param stride: The step size for creating new windows
    :return: A list of dictionaries
    """
    time_windows = []

    for i in range(0, len(time_series) - window_size + 1, stride):
        window_dict = {hour: time_series[i + hour] for hour in range(window_size)}
        time_windows.append(window_dict)

    return time_windows

def create_time_window_dicts_modified(time_series, window_size=6, stride=1):
    """
    Converts a time series into a list of dictionaries using time windows.

    :param time_series: A list of time series values
    :param window_size: The size of the time window (number of hours in each dictionary)
    :param stride: The step size for creating new windows
    :return: A list of dictionaries
    """
    time_windows = []

    # Number of available data points
    series_length = len(time_series)
    
    # Calculate the mean of the time series
    mean_value = sum(time_series) / series_length
    mean_value = round_value(mean_value)
    # Create time windows
    for i in range(0, series_length - window_size + 1, stride):
        window_dict = {hour: round_value(time_series[i + hour]) for hour in range(window_size)}
        time_windows.append(window_dict)
    
    # Complete the time windows for the remaining data with the mean
    for i in range(series_length - window_size + 1, series_length, stride):
        window_dict = {}
        for hour in range(window_size):
            if i + hour < series_length:
                window_dict[hour] = round_value(time_series[i + hour])
            else:
                window_dict[hour] = mean_value
        time_windows.append(window_dict)

    return time_windows

def nlp_opti_model(objective_function_prices,
                   consumption_prediction,
                   wind_prediction,
                   pv_prediction,
                   battery_initial_soc,
                   hydrogen_initial_soc,):
    """
    This function will use to create an optimization block for MPC of Microgrid. It takes predictions of renewable energy
    production and consumption of loads for a specific forecast horizon and then use a Non-linear programming (NLP)
    model to solve the optimization problem and determine the best commands for each component of MG
    Args:
        objective_function_prices: A dictionary that has prices that show the cost of each part of objective function
        consumption_prediction: The prediction that has been made by an external block for load consumption
        wind_prediction: The prediction that has been made by an external block for produced energy by wind turbine
        pv_prediction: The prediction that has been made by an external block for produced energy by pv plant
        battery_initial_soc: The initial soc of battery for this time-step
        hydrogen_initial_soc: The initial soc of hydrogen for this time-step

    Returns: model

    """
    # -------------------------------- Building model ----------------------------------------
    model = pyo.ConcreteModel(name='NLP')
    # -------------------------------- Optimization window ----------------------------------------
    window = len(wind_prediction)
    model.T = pyo.Set(initialize=list(range(window)))
    model.T_1 = pyo.Set(initialize=list(range(window - 1)))
    # -------------------------------- Rounding input values --------------------------------------
    # wind production forecasted for each time step:
    model.p_wind = pyo.Param(model.T,
                             initialize={t: round_value(val) for t, val in wind_prediction.items()})
    # pv production forecasted for each time step:
    model.p_pv = pyo.Param(model.T,
                           initialize={t: round_value(val) for t, val in pv_prediction.items()})
    # loads consumption forecasted for each time step:
    model.p_consumption = pyo.Param(model.T,
                                    initialize={t: round_value(val) for t, val in consumption_prediction.items()})
    # -------------------------------- prices -----------------------------------------------------
    # prices for each part of objective function
    obj_num = len(objective_function_prices)
    model.N = pyo.Set(initialize=list(range(obj_num)))
    model.objective_function_prices = pyo.Param(model.N, initialize=objective_function_prices)
    # -------------------------------- Parameters -------------------------------------------------
    # -------------------------------- BESS ----------------------------------------
    # maximum capacity of battery:
    model.battery_capacity = pyo.Param(initialize=500.)
    # maximum charging rate of battery:
    model.battery_max_charging_rate = pyo.Param(initialize=400.)
    # maximum discharging rate of battery:
    model.battery_max_discharging_rate = pyo.Param(initialize=-400.)
    # charging efficiency of battery:
    model.battery_charging_efficiency = pyo.Param(initialize=0.98)
    # discharging efficiency of battery:
    model.battery_discharging_efficiency = pyo.Param(initialize=0.98)
    # minimum SoC of battery:
    model.battery_min_SoC = pyo.Param(initialize=0.)
    # -------------------------------- HESS ----------------------------------------
    # maximum capacity of hydrogen:
    model.hydrogen_capacity = pyo.Param(initialize=1670.)
    # maximum charging rate of hydrogen:
    model.hydrogen_max_charging_rate = pyo.Param(initialize=55.)
    # maximum discharging rate of hydrogen:
    model.hydrogen_max_discharging_rate = pyo.Param(initialize=-100.)
    # charging efficiency of hydrogen:
    model.hydrogen_charging_efficiency = pyo.Param(initialize=0.64)
    # discharging efficiency of hydrogen:
    model.hydrogen_discharging_efficiency = pyo.Param(initialize=0.50)
    # minimum SoC of hydrogen:
    model.hydrogen_min_SoC = pyo.Param(initialize=0.)
    # -------------------------------- Diesel generator ----------------------------------------
    # maximum power capacity of diesel generator:
    model.diesel_max_power = pyo.Param(initialize=90.)
    # -------------------------------- Loads ---------------------------------------------------
    model.load_max_increase_rate = pyo.Param(initialize=50.)
    model.load_max_decrease_rate = pyo.Param(initialize=-50.)
    # -------------------------------- Variables ----------------------------------------
    # -------------------------------- BESS ----------------------------------------
    # convert power of battery
    model.p_battery = pyo.Var(model.T,
                              bounds=(model.battery_max_discharging_rate, model.battery_max_charging_rate),
                              domain=pyo.Reals)
    # state of charge for battery:
    model.battery_SoC = pyo.Var(model.T,
                                bounds=(model.battery_min_SoC, model.battery_capacity),
                                domain=pyo.NonNegativeReals)
    # -------------------------------- HESS ----------------------------------------
    # convert power of hydrogen
    model.p_hydrogen = pyo.Var(model.T,
                               bounds=(model.hydrogen_max_discharging_rate, model.hydrogen_max_charging_rate),
                               domain=pyo.Reals)
    # state of charge for battery:
    model.hydrogen_SoC = pyo.Var(model.T,
                                bounds=(model.hydrogen_min_SoC, model.hydrogen_capacity),
                                domain=pyo.NonNegativeReals)
    # -------------------------------- Wind ----------------------------------------
    # curtailment power of wind:
    model.p_wind_curtailment = pyo.Var(model.T,
                                       domain=pyo.NonNegativeReals)
    # -------------------------------- PV ----------------------------------------
    # curtailment power of pv:
    model.p_pv_curtailment = pyo.Var(model.T,
                                     domain=pyo.NonNegativeReals)
    # -------------------------------- Diesel generator ----------------------------------------
    # needed power of diesel
    model.p_diesel = pyo.Var(model.T,
                             bounds=(0, model.diesel_max_power),
                             domain=pyo.NonNegativeReals)
    # -------------------------------- Load ----------------------------------------
    # power change command of load
    model.p_load_change = pyo.Var(model.T,
                                      bounds=(model.load_max_decrease_rate, model.load_max_increase_rate),
                                      domain=pyo.Reals)
    # -----------------------------load Auxiliary variables ------------------------------------------------------
    model.p_load_grow = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_load_shed = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # ----------------------------load Auxiliary constraints ----------------------------------------------------
    def grow_shed_rule(mdl, t):
        return mdl.p_load_change[t] == mdl.p_load_grow[t] - mdl.p_load_shed[t]

    def grow_shed_non_negative_rule(mdl, t):
        return mdl.p_load_grow[t] * mdl.p_load_shed[t] == 0

    model.constr_grow_shed = pyo.Constraint(model.T, rule=grow_shed_rule)
    model.constr_grow_shed_non_negative = pyo.Constraint(model.T, rule=grow_shed_non_negative_rule)

    #-----------------------------Battery Auxiliary variables ------------------------------------------------------
    model.p_battery_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_battery_discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    #----------------------------Battery Auxiliary constraints ----------------------------------------------------
    def battery_charge_discharge_rule(mdl, t):
        return  mdl.p_battery[t] == mdl.p_battery_charge[t] - mdl.p_battery_discharge[t]

    def battery_charge_discharge_non_negative_rule(mdl, t):
        return mdl.p_battery_charge[t] * mdl.p_battery_discharge[t] == 0

    model.constr_battery_charge_discharge = pyo.Constraint(model.T, rule=battery_charge_discharge_rule)
    model.constr_battery_charge_discharge_non_negative = pyo.Constraint(model.T,
                                                                         rule=battery_charge_discharge_non_negative_rule)

    # -----------------------------hydrogen Auxiliary variables ------------------------------------------------------
    model.p_hydrogen_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_hydrogen_discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # ----------------------------hydrogen Auxiliary constraints ----------------------------------------------------
    def hydrogen_charge_discharge_rule(mdl, t):
        return mdl.p_hydrogen[t] == mdl.p_hydrogen_charge[t] - mdl.p_hydrogen_discharge[t]

    def hydrogen_charge_discharge_non_negative_rule(mdl, t):
        return  mdl.p_hydrogen_charge[t] * mdl.p_hydrogen_discharge[t] == 0

    model.constr_hydrogen_charge_discharge = pyo.Constraint(model.T, rule=hydrogen_charge_discharge_rule)
    model.constr_hydrogen_charge_discharge_non_negative = pyo.Constraint(model.T,
                                                                         rule=hydrogen_charge_discharge_non_negative_rule)
    # -------------------------------- Constraints ----------------------------------------
    # -------------------------------- Battery recursive SoC equation ----------------------------------------
    # Recursive SoC equation of battery:
    def constr_battery_recursive_soc_rule(mdl, t):
        if t == 0:
            return mdl.battery_SoC[t] == battery_initial_soc
        else:
            return (mdl.battery_SoC[t] ==
                    mdl.battery_SoC[t - 1] +
                    mdl.battery_charging_efficiency *  mdl.p_battery_charge[t - 1] -
                    mdl.p_battery_discharge[t - 1] / mdl.battery_discharging_efficiency)

    model.constr_battery_recursive_soc = pyo.Constraint(model.T, rule=constr_battery_recursive_soc_rule)

    # -------------------------------- hydrogen recursive SoC equation ----------------------------------------
    # Recursive SoC equation of hydrogen:
    def constr_hydrogen_recursive_soc_rule(mdl, t):
        if t == 0:
            return mdl.hydrogen_SoC[t] == hydrogen_initial_soc
        else:
            return (mdl.hydrogen_SoC[t] ==
                    mdl.hydrogen_SoC[t - 1] +
                    mdl.hydrogen_charging_efficiency * mdl.p_hydrogen_charge[t - 1] -
                    mdl.p_hydrogen_discharge[t - 1] / mdl.hydrogen_discharging_efficiency)

    model.constr_hydrogen_recursive_soc = pyo.Constraint(model.T, rule=constr_hydrogen_recursive_soc_rule)

    # -------------------------------- Power equality ----------------------------------------
    # In each time step 't' sum of generation and sum of consumption must be equal
    def constr_power_equality_rule(mdl, t):
        return (mdl.p_diesel[t] +
                mdl.p_wind[t] +
                mdl.p_pv[t]
                ==
                mdl.p_battery[t] +
                mdl.p_hydrogen[t] +
                mdl.p_wind_curtailment[t] +
                mdl.p_pv_curtailment[t] +
                mdl.p_consumption[t] +
                mdl.p_load_change[t])

    model.constr_power_equality = pyo.Constraint(model.T, rule=constr_power_equality_rule)

    # -------------------------------- Wind curtailment ----------------------------------------
    # At each time step [t] curtailment power of wind generation must be less than or equal to wind generation:
    def constr_wind_curtailment_rule(mdl, t):
        return mdl.p_wind_curtailment[t] <= mdl.p_wind[t]

    model.constr_wind_curtailment = pyo.Constraint(model.T, rule=constr_wind_curtailment_rule)

    # -------------------------------- PV curtailment ----------------------------------------
    # At each time step [t] curtailment power of pv generation must be less than or equal to pv generation:
    def constr_pv_curtailment_rule(mdl, t):
        return mdl.p_pv_curtailment[t] <= mdl.p_pv[t]

    model.constr_pv_curtailment = pyo.Constraint(model.T, rule=constr_pv_curtailment_rule)

    # -------------------------------- Objective Function ----------------------------------------

    # def objective_function(mdl):
    #     f_load = (mdl.objective_function_prices[0] * sum([mdl.p_load_grow[t] for t in mdl.T]) +
    #               mdl.objective_function_prices[1] * sum([mdl.p_load_shed[t] for t in mdl.T]))
    #     # diesel generation over window period have to be minimized:
    #     f_diesel = mdl.objective_function_prices[2] * sum([mdl.p_diesel[t] for t in mdl.T])
    #     # curtailed power of wind generation over window period have to be minimized:
    #     f_wind_curtailed = mdl.objective_function_prices[3] * sum([mdl.p_wind_curtailment[t] for t in mdl.T])
    #     # curtailed power of pv generation over window period have to be minimum:
    #     f_pv_curtailed = mdl.objective_function_prices[4] * sum([mdl.p_pv_curtailment[t] for t in mdl.T])
    #     # charge and discharge loss of battery have to be minimum:
    #     f_battery_loss = (mdl.objective_function_prices[5] *
    #                       (((1 - mdl.battery_discharging_efficiency) / mdl.battery_discharging_efficiency) *
    #                       sum([mdl.p_battery_discharge[t] for t in mdl.T]) +
    #                       (1 - mdl.battery_charging_efficiency) *
    #                       sum([mdl.p_battery_charge[t] for t in mdl.T])))
    #     # charge and discharge loss of hydrogen have to be minimum:
    #     f_hydrogen_loss = (mdl.objective_function_prices[6] *
    #                        (((1 - mdl.hydrogen_discharging_efficiency) / mdl.hydrogen_discharging_efficiency) *
    #                        sum([mdl.p_hydrogen_discharge[t] for t in mdl.T]) +
    #                        (1 - mdl.hydrogen_charging_efficiency) *
    #                        sum([mdl.p_hydrogen_charge[t] for t in mdl.T])))
    #
    #     return  f_load + f_diesel + f_wind_curtailed + f_pv_curtailed + f_battery_loss + f_hydrogen_loss
    #
    # model.objective = pyo.Objective(rule=objective_function)

    def objective_function(mdl):
        # Load change cost
        f_load = (mdl.objective_function_prices[0] * sum(mdl.p_load_grow[t] for t in mdl.T) +
                  mdl.objective_function_prices[1] * sum(mdl.p_load_shed[t] for t in mdl.T))

        # Diesel generation cost
        f_diesel = mdl.objective_function_prices[2] * sum(mdl.p_diesel[t] for t in mdl.T)

        # Curtailment costs
        f_wind_curtailed = mdl.objective_function_prices[3] * sum(mdl.p_wind_curtailment[t] for t in mdl.T)
        f_pv_curtailed = mdl.objective_function_prices[4] * sum(mdl.p_pv_curtailment[t] for t in mdl.T)

        # Battery loss cost
        f_battery_loss = mdl.objective_function_prices[5] * (
                ((1 - mdl.battery_discharging_efficiency) / mdl.battery_discharging_efficiency) *
                sum(mdl.p_battery_discharge[t] for t in mdl.T) +
                (1 - mdl.battery_charging_efficiency) *
                sum(mdl.p_battery_charge[t] for t in mdl.T)
        )

        # Hydrogen loss cost
        f_hydrogen_loss = mdl.objective_function_prices[6] * (
                ((1 - mdl.hydrogen_discharging_efficiency) / mdl.hydrogen_discharging_efficiency) *
                sum(mdl.p_hydrogen_discharge[t] for t in mdl.T) +
                (1 - mdl.hydrogen_charging_efficiency) *
                sum(mdl.p_hydrogen_charge[t] for t in mdl.T)
        )

        return f_load + f_diesel + f_wind_curtailed + f_pv_curtailed + f_battery_loss + f_hydrogen_loss

    # افزودن تابع هدف به مدل
    model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    return model

def milp_opti_model(objective_function_prices,
                   consumption_prediction,
                   wind_prediction,
                   pv_prediction,
                   battery_initial_soc,
                   hydrogen_initial_soc,):
    """
    This function will use to create an optimization block for MPC of Microgrid. It takes predictions of renewable energy
    production and consumption of loads for a specific forecast horizon and then use a mixed_integer-linear_programming
    (MILP) model to solve the optimization problem and determine the best control_commands for each component of MG
    Args:
        objective_function_prices: A dictionary that has prices that show the cost of each part of objective function
        consumption_prediction: The prediction that has been made by an external block for load consumption
        wind_prediction: The prediction that has been made by an external block for produced energy by wind turbine
        pv_prediction: The prediction that has been made by an external block for produced energy by pv plant
        battery_initial_soc: The initial soc of battery for this time-step
        hydrogen_initial_soc: The initial soc of hydrogen for this time-step

    Returns: model

    """
    # -------------------------------- Building model ----------------------------------------
    model = pyo.ConcreteModel(name='MILP')
    # -------------------------------- Optimization window ----------------------------------------
    window = len(wind_prediction)
    model.T = pyo.Set(initialize=list(range(window)))
    # -------------------------------- Rounding input values --------------------------------------
    # wind production forecasted for each time step:
    model.p_wind = pyo.Param(model.T,
                             initialize={t: round_value(val) for t, val in wind_prediction.items()})
    # pv production forecasted for each time step:
    model.p_pv = pyo.Param(model.T,
                           initialize={t: round_value(val) for t, val in pv_prediction.items()})
    # loads consumption forecasted for each time step:
    model.p_consumption = pyo.Param(model.T,
                                    initialize={t: round_value(val) for t, val in consumption_prediction.items()})
    # -------------------------------- prices -----------------------------------------------------
    # prices for each part of objective function
    obj_num = len(objective_function_prices)
    model.N = pyo.Set(initialize=list(range(obj_num)))
    model.objective_function_prices = pyo.Param(model.N, initialize=objective_function_prices)
    # -------------------------------- Parameters -------------------------------------------------
    # -------------------------------- BESS ----------------------------------------
    # maximum capacity of battery:
    model.battery_capacity = pyo.Param(initialize=500.)
    # maximum charging rate of battery:
    model.battery_max_charging_rate = pyo.Param(initialize=400.)
    # maximum discharging rate of battery:
    model.battery_max_discharging_rate = pyo.Param(initialize=400.)
    # charging efficiency of battery:
    model.battery_charging_efficiency = pyo.Param(initialize=0.98)
    # discharging efficiency of battery:
    model.battery_discharging_efficiency = pyo.Param(initialize=0.98)
    # minimum SoC of battery:
    model.battery_min_SoC = pyo.Param(initialize=0.)
    # -------------------------------- HESS ----------------------------------------
    # maximum capacity of hydrogen:
    model.hydrogen_capacity = pyo.Param(initialize=1670.)
    # maximum charging rate of hydrogen:
    model.hydrogen_max_charging_rate = pyo.Param(initialize=55.)
    # maximum discharging rate of hydrogen:
    model.hydrogen_max_discharging_rate = pyo.Param(initialize=100.)
    # charging efficiency of hydrogen:
    model.hydrogen_charging_efficiency = pyo.Param(initialize=0.64)
    # discharging efficiency of hydrogen:
    model.hydrogen_discharging_efficiency = pyo.Param(initialize=0.50)
    # minimum SoC of hydrogen:
    model.hydrogen_min_SoC = pyo.Param(initialize=0.)
    # -------------------------------- Diesel generator ----------------------------------------
    # maximum power capacity of diesel generator:
    model.diesel_max_power = pyo.Param(initialize=90.)
    # -------------------------------- Loads ---------------------------------------------------
    model.load_max_increase_rate = pyo.Param(initialize=50.)
    model.load_max_decrease_rate = pyo.Param(initialize=50.)
    # -------------------------------- Variables ----------------------------------------
    # -------------------------------- BESS ----------------------------------------

    model.p_battery_charge = pyo.Var(model.T,
                                     domain=pyo.NonNegativeReals)

    model.p_battery_discharge = pyo.Var(model.T,
                                        domain=pyo.NonNegativeReals)

    model.z_battery = pyo.Var(model.T,
                              domain=pyo.Binary)  # 1 for charge and 0 for discharge
    # state of charge for battery:
    model.battery_SoC = pyo.Var(model.T,
                                bounds=(model.battery_min_SoC, model.battery_capacity),
                                domain=pyo.NonNegativeReals)
    # -------------------------------- HESS ----------------------------------------

    model.p_hydrogen_charge = pyo.Var(model.T,
                                      domain=pyo.NonNegativeReals)

    model.p_hydrogen_discharge = pyo.Var(model.T,
                                         domain=pyo.NonNegativeReals)

    model.z_hydrogen = pyo.Var(model.T,
                               domain=pyo.Binary)  # 1 for charge and 0 for discharge
    # state of charge for battery:
    model.hydrogen_SoC = pyo.Var(model.T,
                                bounds=(model.hydrogen_min_SoC, model.hydrogen_capacity),
                                domain=pyo.NonNegativeReals)
    # -------------------------------- Wind ----------------------------------------
    # curtailment power of wind:
    model.p_wind_curtailment = pyo.Var(model.T,
                                       domain=pyo.NonNegativeReals)
    # -------------------------------- PV ----------------------------------------
    # curtailment power of pv:
    model.p_pv_curtailment = pyo.Var(model.T,
                                     domain=pyo.NonNegativeReals)
    # -------------------------------- Diesel generator ----------------------------------------
    # needed power of diesel
    model.p_diesel = pyo.Var(model.T,
                             bounds=(0, model.diesel_max_power),
                             domain=pyo.NonNegativeReals)
    # -------------------------------- Load ----------------------------------------

    model.p_load_grow = pyo.Var(model.T,
                                domain=pyo.NonNegativeReals)

    model.p_load_shed = pyo.Var(model.T,
                                domain=pyo.NonNegativeReals,)

    model.z_load = pyo.Var(model.T,
                           domain=pyo.Binary) # 1 for grow and 0 for shed

    # -------------------------------- Constraints -------------------------------------------------------------
    # ----------------------------load Auxiliary constraints ----------------------------------------------------

    def asynchronous_grow_rule(mdl, t):
        return mdl.p_load_grow[t] <= mdl.z_load[t] * mdl.load_max_increase_rate

    model.constr_asynchronous_grow = pyo.Constraint(model.T,
                                                    rule=asynchronous_grow_rule)

    def asynchronous_shed_rule(mdl, t):
        return mdl.p_load_shed[t] <= (1 - mdl.z_load[t]) * mdl.load_max_decrease_rate

    model.constr_asynchronous_shed = pyo.Constraint(model.T,
                                                    rule=asynchronous_shed_rule)

    #----------------------------Battery Auxiliary constraints ----------------------------------------------------

    def asynchronous_battery_charge_rule(mdl, t):
        return mdl.p_battery_charge[t] <= mdl.z_battery[t] * mdl.battery_max_charging_rate

    model.constr_asynchronous_battery_charge = pyo.Constraint(model.T,
                                                    rule=asynchronous_battery_charge_rule)

    def asynchronous_battery_discharge_rule(mdl, t):
        return mdl.p_battery_discharge[t] <= (1 - mdl.z_battery[t]) * mdl.battery_max_discharging_rate

    model.constr_asynchronous_battery_discharge = pyo.Constraint(model.T,
                                                    rule=asynchronous_battery_discharge_rule)

    def battery_next_soc_less_than_capacity_rule(mdl, t):
        return (mdl.battery_SoC[t] + mdl.p_battery_charge[t] * mdl.battery_charging_efficiency <=
                mdl.battery_capacity)

    model.constr_battery_next_soc_less_than_capacity = pyo.Constraint(model.T,
                                                              rule=battery_next_soc_less_than_capacity_rule)

    def battery_next_soc_greater_than_min_soc_rule(mdl, t):
        return (mdl.battery_SoC[t] - mdl.p_battery_discharge[t] / mdl.battery_discharging_efficiency >=
                mdl.battery_min_SoC)

    model.constr_battery_next_soc_greater_than_min_soc = pyo.Constraint(model.T,
                                                                rule=battery_next_soc_greater_than_min_soc_rule)

    # ----------------------------hydrogen Auxiliary constraints ----------------------------------------------------

    def asynchronous_hydrogen_charge_rule(mdl, t):
        return mdl.p_hydrogen_charge[t] <= mdl.z_hydrogen[t] * mdl.hydrogen_max_charging_rate

    model.constr_asynchronous_hydrogen_charge = pyo.Constraint(model.T,
                                                    rule=asynchronous_hydrogen_charge_rule)

    def asynchronous_hydrogen_discharge_rule(mdl, t):
        return mdl.p_hydrogen_discharge[t] <= (1 - mdl.z_hydrogen[t]) * mdl.hydrogen_max_discharging_rate

    model.constr_asynchronous_hydrogen_discharge = pyo.Constraint(model.T,
                                                    rule=asynchronous_hydrogen_discharge_rule)

    def hydrogen_next_soc_less_than_capacity_rule(mdl, t):
        return (mdl.hydrogen_SoC[t] + mdl.p_hydrogen_charge[t] * mdl.hydrogen_charging_efficiency <=
                mdl.hydrogen_capacity)

    model.constr_hydrogen_next_soc_less_than_capacity = pyo.Constraint(model.T,
                                                              rule=hydrogen_next_soc_less_than_capacity_rule)

    def hydrogen_next_soc_greater_than_min_soc_rule(mdl, t):
        return (mdl.hydrogen_SoC[t] - mdl.p_hydrogen_discharge[t] / mdl.hydrogen_discharging_efficiency >=
                mdl.hydrogen_min_SoC)

    model.constr_hydrogen_next_soc_greater_than_min_soc = pyo.Constraint(model.T,
                                                                rule=hydrogen_next_soc_greater_than_min_soc_rule)

    # -------------------------------- Battery recursive SoC equation ----------------------------------------
    # Recursive SoC equation of battery:
    def constr_battery_recursive_soc_rule(mdl, t):
        if t == 0:
            return mdl.battery_SoC[t] == battery_initial_soc
        else:
            return (mdl.battery_SoC[t] ==
                    mdl.battery_SoC[t - 1] +
                    mdl.battery_charging_efficiency *  mdl.p_battery_charge[t - 1] -
                    mdl.p_battery_discharge[t - 1] / mdl.battery_discharging_efficiency)

    model.constr_battery_recursive_soc = pyo.Constraint(model.T,
                                                        rule=constr_battery_recursive_soc_rule)

    # -------------------------------- hydrogen recursive SoC equation ----------------------------------------
    # Recursive SoC equation of hydrogen:
    def constr_hydrogen_recursive_soc_rule(mdl, t):
        if t == 0:
            return mdl.hydrogen_SoC[t] == hydrogen_initial_soc
        else:
            return (mdl.hydrogen_SoC[t] ==
                    mdl.hydrogen_SoC[t - 1] +
                    mdl.hydrogen_charging_efficiency * mdl.p_hydrogen_charge[t - 1] -
                    mdl.p_hydrogen_discharge[t - 1] / mdl.hydrogen_discharging_efficiency)

    model.constr_hydrogen_recursive_soc = pyo.Constraint(model.T,
                                                         rule=constr_hydrogen_recursive_soc_rule)

    # -------------------------------- Power equality ----------------------------------------
    # In each time step 't' sum of generation and sum of consumption must be equal
    def constr_power_equality_rule(mdl, t):
        return (mdl.p_diesel[t] +
                mdl.p_wind[t] +
                mdl.p_pv[t] +
                mdl.p_battery_discharge[t] +
                mdl.p_hydrogen_discharge[t] +
                mdl.p_load_shed[t]
                ==
                mdl.p_battery_charge[t] +
                mdl.p_hydrogen_charge[t] +
                mdl.p_wind_curtailment[t] +
                mdl.p_pv_curtailment[t] +
                mdl.p_consumption[t] +
                mdl.p_load_grow[t])

    model.constr_power_equality = pyo.Constraint(model.T,
                                                 rule=constr_power_equality_rule)

    # -------------------------------- Wind curtailment ----------------------------------------
    # At each time step [t] curtailment power of wind generation must be less than or equal to wind generation:
    def constr_wind_curtailment_rule(mdl, t):
        return mdl.p_wind_curtailment[t] <= mdl.p_wind[t]

    model.constr_wind_curtailment = pyo.Constraint(model.T,
                                                   rule=constr_wind_curtailment_rule)

    # -------------------------------- PV curtailment ----------------------------------------
    # At each time step [t] curtailment power of pv generation must be less than or equal to pv generation:
    def constr_pv_curtailment_rule(mdl, t):
        return mdl.p_pv_curtailment[t] <= mdl.p_pv[t]

    model.constr_pv_curtailment = pyo.Constraint(model.T,
                                                 rule=constr_pv_curtailment_rule)

    # -------------------------------- Objective Function ----------------------------------------
    def objective_function(mdl):
        # Load change cost
        f_load = (mdl.objective_function_prices[0] * sum(mdl.p_load_grow[t] for t in mdl.T) +
                  mdl.objective_function_prices[1] * sum(mdl.p_load_shed[t] for t in mdl.T))

        # Diesel generation cost
        f_diesel = mdl.objective_function_prices[2] * sum(mdl.p_diesel[t] for t in mdl.T)

        # Curtailment costs
        f_wind_curtailed = mdl.objective_function_prices[3] * sum(mdl.p_wind_curtailment[t] for t in mdl.T)
        f_pv_curtailed = mdl.objective_function_prices[4] * sum(mdl.p_pv_curtailment[t] for t in mdl.T)

        # Battery loss cost
        f_battery_loss = mdl.objective_function_prices[5] * (
                ((1 - mdl.battery_discharging_efficiency) / mdl.battery_discharging_efficiency) *
                sum(mdl.p_battery_discharge[t] for t in mdl.T) +
                (1 - mdl.battery_charging_efficiency) *
                sum(mdl.p_battery_charge[t] for t in mdl.T)
        )

        # Hydrogen loss cost
        f_hydrogen_loss = mdl.objective_function_prices[6] * (
                ((1 - mdl.hydrogen_discharging_efficiency) / mdl.hydrogen_discharging_efficiency) *
                sum(mdl.p_hydrogen_discharge[t] for t in mdl.T) +
                (1 - mdl.hydrogen_charging_efficiency) *
                sum(mdl.p_hydrogen_charge[t] for t in mdl.T)
        )

        return f_load + f_diesel + f_wind_curtailed + f_pv_curtailed + f_battery_loss + f_hydrogen_loss

    # Add the objective function to the model
    model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    return model


def milp_opti_model_connected(objective_function_prices,
                   consumption_prediction,
                   wind_prediction,
                   pv_prediction,
                   battery_initial_soc,
                   hydrogen_initial_soc,):
    """
    This function will use to create an optimization block for MPC of Microgrid. It takes predictions of renewable energy
    production and consumption of loads for a specific forecast horizon and then use a mixed_integer-linear_programming
    (MILP) model to solve the optimization problem and determine the best control_commands for each component of MG
    Args:
        objective_function_prices: A dictionary that has prices that show the cost of each part of objective function
        consumption_prediction: The prediction that has been made by an external block for load consumption
        wind_prediction: The prediction that has been made by an external block for produced energy by wind turbine
        pv_prediction: The prediction that has been made by an external block for produced energy by pv plant
        battery_initial_soc: The initial soc of battery for this time-step
        hydrogen_initial_soc: The initial soc of hydrogen for this time-step

    Returns: model

    """
    # -------------------------------- Building model ----------------------------------------
    model = pyo.ConcreteModel(name='MILP')
    # -------------------------------- Optimization window ----------------------------------------
    window = len(wind_prediction)
    model.T = pyo.Set(initialize=list(range(window)))
    # -------------------------------- Rounding input values --------------------------------------
    # wind production forecasted for each time step:
    model.p_wind = pyo.Param(model.T,
                             initialize={t: round_value(val) for t, val in wind_prediction.items()})
    # pv production forecasted for each time step:
    model.p_pv = pyo.Param(model.T,
                           initialize={t: round_value(val) for t, val in pv_prediction.items()})
    # loads consumption forecasted for each time step:
    model.p_consumption = pyo.Param(model.T,
                                    initialize={t: round_value(val) for t, val in consumption_prediction.items()})
    # -------------------------------- prices -----------------------------------------------------
    # prices for each part of objective function
    obj_num = len(objective_function_prices)
    model.N = pyo.Set(initialize=list(range(obj_num)))
    model.objective_function_prices = pyo.Param(model.N, initialize=objective_function_prices)
    # -------------------------------- Parameters -------------------------------------------------
    # -------------------------------- BESS ----------------------------------------
    # maximum capacity of battery:
    model.battery_capacity = pyo.Param(initialize=500.)
    # maximum charging rate of battery:
    model.battery_max_charging_rate = pyo.Param(initialize=400.)
    # maximum discharging rate of battery:
    model.battery_max_discharging_rate = pyo.Param(initialize=400.)
    # charging efficiency of battery:
    model.battery_charging_efficiency = pyo.Param(initialize=0.98)
    # discharging efficiency of battery:
    model.battery_discharging_efficiency = pyo.Param(initialize=0.98)
    # minimum SoC of battery:
    model.battery_min_SoC = pyo.Param(initialize=0.)
    # -------------------------------- HESS ----------------------------------------
    # maximum capacity of hydrogen:
    model.hydrogen_capacity = pyo.Param(initialize=1670.)
    # maximum charging rate of hydrogen:
    model.hydrogen_max_charging_rate = pyo.Param(initialize=55.)
    # maximum discharging rate of hydrogen:
    model.hydrogen_max_discharging_rate = pyo.Param(initialize=100.)
    # charging efficiency of hydrogen:
    model.hydrogen_charging_efficiency = pyo.Param(initialize=0.64)
    # discharging efficiency of hydrogen:
    model.hydrogen_discharging_efficiency = pyo.Param(initialize=0.50)
    # minimum SoC of hydrogen:
    model.hydrogen_min_SoC = pyo.Param(initialize=0.)
    # -------------------------------- Diesel generator ----------------------------------------
    # maximum power capacity of diesel generator:
    model.diesel_max_power = pyo.Param(initialize=90.)
    # -------------------------------- Loads ---------------------------------------------------
    model.load_max_increase_rate = pyo.Param(initialize=50.)
    model.load_max_decrease_rate = pyo.Param(initialize=50.)
    # -------------------------------- Exgrid ---------------------------------------------------
    model.exgrid_max_import = pyo.Param(initialize=100.)
    model.exgrid_max_export = pyo.Param(initialize=100.)
    # -------------------------------- Variables ----------------------------------------
    # -------------------------------- BESS ----------------------------------------

    model.p_battery_charge = pyo.Var(model.T,
                                     domain=pyo.NonNegativeReals)

    model.p_battery_discharge = pyo.Var(model.T,
                                        domain=pyo.NonNegativeReals)

    model.z_battery = pyo.Var(model.T,
                              domain=pyo.Binary)  # 1 for charge and 0 for discharge
    # state of charge for battery:
    model.battery_SoC = pyo.Var(model.T,
                                bounds=(model.battery_min_SoC, model.battery_capacity),
                                domain=pyo.NonNegativeReals)
    # -------------------------------- HESS ----------------------------------------

    model.p_hydrogen_charge = pyo.Var(model.T,
                                      domain=pyo.NonNegativeReals)

    model.p_hydrogen_discharge = pyo.Var(model.T,
                                         domain=pyo.NonNegativeReals)

    model.z_hydrogen = pyo.Var(model.T,
                               domain=pyo.Binary)  # 1 for charge and 0 for discharge
    # state of charge for battery:
    model.hydrogen_SoC = pyo.Var(model.T,
                                bounds=(model.hydrogen_min_SoC, model.hydrogen_capacity),
                                domain=pyo.NonNegativeReals)
    # -------------------------------- Wind ----------------------------------------
    # curtailment power of wind:
    model.p_wind_curtailment = pyo.Var(model.T,
                                       domain=pyo.NonNegativeReals)
    # -------------------------------- PV ----------------------------------------
    # curtailment power of pv:
    model.p_pv_curtailment = pyo.Var(model.T,
                                     domain=pyo.NonNegativeReals)
    # -------------------------------- Diesel generator ----------------------------------------
    # needed power of diesel
    model.p_diesel = pyo.Var(model.T,
                             bounds=(0, model.diesel_max_power),
                             domain=pyo.NonNegativeReals)
    # -------------------------------- Load ----------------------------------------

    model.p_load_grow = pyo.Var(model.T,
                                domain=pyo.NonNegativeReals)

    model.p_load_shed = pyo.Var(model.T,
                                domain=pyo.NonNegativeReals,)

    model.z_load = pyo.Var(model.T,
                           domain=pyo.Binary) # 1 for grow and 0 for shed
# -------------------------------- Exgrid ----------------------------------------

    model.p_exgrid_export = pyo.Var(model.T,
                                domain=pyo.NonNegativeReals)

    model.p_exgrid_import = pyo.Var(model.T,
                                domain=pyo.NonNegativeReals,)

    model.z_exgrid = pyo.Var(model.T,
                           domain=pyo.Binary) # 1 for export and 0 for import

    # -------------------------------- Constraints -------------------------------------------------------------
      # ----------------------------Exgrid Auxiliary constraints ----------------------------------------------------

    def asynchronous_export_rule(mdl, t):
        return mdl.p_exgrid_export[t] <= mdl.z_exgrid[t] * mdl.exgrid_max_export

    model.constr_asynchronous_export = pyo.Constraint(model.T,
                                                    rule=asynchronous_export_rule)

    def asynchronous_import_rule(mdl, t):
        return mdl.p_exgrid_import[t] <= (1 - mdl.z_exgrid[t]) * mdl.exgrid_max_import

    model.constr_asynchronous_import = pyo.Constraint(model.T,
                                                    rule=asynchronous_import_rule)

    # ----------------------------load Auxiliary constraints ----------------------------------------------------

    def asynchronous_grow_rule(mdl, t):
        return mdl.p_load_grow[t] <= mdl.z_load[t] * mdl.load_max_increase_rate

    model.constr_asynchronous_grow = pyo.Constraint(model.T,
                                                    rule=asynchronous_grow_rule)

    def asynchronous_shed_rule(mdl, t):
        return mdl.p_load_shed[t] <= (1 - mdl.z_load[t]) * mdl.load_max_decrease_rate

    model.constr_asynchronous_shed = pyo.Constraint(model.T,
                                                    rule=asynchronous_shed_rule)

    #----------------------------Battery Auxiliary constraints ----------------------------------------------------

    def asynchronous_battery_charge_rule(mdl, t):
        return mdl.p_battery_charge[t] <= mdl.z_battery[t] * mdl.battery_max_charging_rate

    model.constr_asynchronous_battery_charge = pyo.Constraint(model.T,
                                                    rule=asynchronous_battery_charge_rule)

    def asynchronous_battery_discharge_rule(mdl, t):
        return mdl.p_battery_discharge[t] <= (1 - mdl.z_battery[t]) * mdl.battery_max_discharging_rate

    model.constr_asynchronous_battery_discharge = pyo.Constraint(model.T,
                                                    rule=asynchronous_battery_discharge_rule)

    def battery_next_soc_less_than_capacity_rule(mdl, t):
        return (mdl.battery_SoC[t] + mdl.p_battery_charge[t] * mdl.battery_charging_efficiency <=
                mdl.battery_capacity)

    model.constr_battery_next_soc_less_than_capacity = pyo.Constraint(model.T,
                                                              rule=battery_next_soc_less_than_capacity_rule)

    def battery_next_soc_greater_than_min_soc_rule(mdl, t):
        return (mdl.battery_SoC[t] - mdl.p_battery_discharge[t] / mdl.battery_discharging_efficiency >=
                mdl.battery_min_SoC)

    model.constr_battery_next_soc_greater_than_min_soc = pyo.Constraint(model.T,
                                                                rule=battery_next_soc_greater_than_min_soc_rule)

    # ----------------------------hydrogen Auxiliary constraints ----------------------------------------------------

    def asynchronous_hydrogen_charge_rule(mdl, t):
        return mdl.p_hydrogen_charge[t] <= mdl.z_hydrogen[t] * mdl.hydrogen_max_charging_rate

    model.constr_asynchronous_hydrogen_charge = pyo.Constraint(model.T,
                                                    rule=asynchronous_hydrogen_charge_rule)

    def asynchronous_hydrogen_discharge_rule(mdl, t):
        return mdl.p_hydrogen_discharge[t] <= (1 - mdl.z_hydrogen[t]) * mdl.hydrogen_max_discharging_rate

    model.constr_asynchronous_hydrogen_discharge = pyo.Constraint(model.T,
                                                    rule=asynchronous_hydrogen_discharge_rule)

    def hydrogen_next_soc_less_than_capacity_rule(mdl, t):
        return (mdl.hydrogen_SoC[t] + mdl.p_hydrogen_charge[t] * mdl.hydrogen_charging_efficiency <=
                mdl.hydrogen_capacity)

    model.constr_hydrogen_next_soc_less_than_capacity = pyo.Constraint(model.T,
                                                              rule=hydrogen_next_soc_less_than_capacity_rule)

    def hydrogen_next_soc_greater_than_min_soc_rule(mdl, t):
        return (mdl.hydrogen_SoC[t] - mdl.p_hydrogen_discharge[t] / mdl.hydrogen_discharging_efficiency >=
                mdl.hydrogen_min_SoC)

    model.constr_hydrogen_next_soc_greater_than_min_soc = pyo.Constraint(model.T,
                                                                rule=hydrogen_next_soc_greater_than_min_soc_rule)

    # -------------------------------- Battery recursive SoC equation ----------------------------------------
    # Recursive SoC equation of battery:
    def constr_battery_recursive_soc_rule(mdl, t):
        if t == 0:
            return mdl.battery_SoC[t] == battery_initial_soc
        else:
            return (mdl.battery_SoC[t] ==
                    mdl.battery_SoC[t - 1] +
                    mdl.battery_charging_efficiency *  mdl.p_battery_charge[t - 1] -
                    mdl.p_battery_discharge[t - 1] / mdl.battery_discharging_efficiency)

    model.constr_battery_recursive_soc = pyo.Constraint(model.T,
                                                        rule=constr_battery_recursive_soc_rule)

    # -------------------------------- hydrogen recursive SoC equation ----------------------------------------
    # Recursive SoC equation of hydrogen:
    def constr_hydrogen_recursive_soc_rule(mdl, t):
        if t == 0:
            return mdl.hydrogen_SoC[t] == hydrogen_initial_soc
        else:
            return (mdl.hydrogen_SoC[t] ==
                    mdl.hydrogen_SoC[t - 1] +
                    mdl.hydrogen_charging_efficiency * mdl.p_hydrogen_charge[t - 1] -
                    mdl.p_hydrogen_discharge[t - 1] / mdl.hydrogen_discharging_efficiency)

    model.constr_hydrogen_recursive_soc = pyo.Constraint(model.T,
                                                         rule=constr_hydrogen_recursive_soc_rule)

    # -------------------------------- Power equality ----------------------------------------
    # In each time step 't' sum of generation and sum of consumption must be equal
    def constr_power_equality_rule(mdl, t):
        return (mdl.p_diesel[t] +
                mdl.p_wind[t] +
                mdl.p_pv[t] +
                mdl.p_battery_discharge[t] +
                mdl.p_hydrogen_discharge[t] +
                mdl.p_load_shed[t] +
                mdl.p_exgrid_import[t]
                ==
                mdl.p_battery_charge[t] +
                mdl.p_hydrogen_charge[t] +
                mdl.p_wind_curtailment[t] +
                mdl.p_pv_curtailment[t] +
                mdl.p_consumption[t] +
                mdl.p_load_grow[t] +
                mdl.p_exgrid_export[t])

    model.constr_power_equality = pyo.Constraint(model.T,
                                                 rule=constr_power_equality_rule)

    # -------------------------------- Wind curtailment ----------------------------------------
    # At each time step [t] curtailment power of wind generation must be less than or equal to wind generation:
    def constr_wind_curtailment_rule(mdl, t):
        return mdl.p_wind_curtailment[t] <= mdl.p_wind[t]

    model.constr_wind_curtailment = pyo.Constraint(model.T,
                                                   rule=constr_wind_curtailment_rule)

    # -------------------------------- PV curtailment ----------------------------------------
    # At each time step [t] curtailment power of pv generation must be less than or equal to pv generation:
    def constr_pv_curtailment_rule(mdl, t):
        return mdl.p_pv_curtailment[t] <= mdl.p_pv[t]

    model.constr_pv_curtailment = pyo.Constraint(model.T,
                                                 rule=constr_pv_curtailment_rule)

    # -------------------------------- Objective Function ----------------------------------------
    def objective_function(mdl):
        # Load change cost
        f_load = (mdl.objective_function_prices[0] * sum(mdl.p_load_grow[t] for t in mdl.T) +
                  mdl.objective_function_prices[1] * sum(mdl.p_load_shed[t] for t in mdl.T))

        # Diesel generation cost
        f_diesel = mdl.objective_function_prices[2] * sum(mdl.p_diesel[t] for t in mdl.T)

        # Curtailment costs
        f_wind_curtailed = mdl.objective_function_prices[3] * sum(mdl.p_wind_curtailment[t] for t in mdl.T)
        f_pv_curtailed = mdl.objective_function_prices[4] * sum(mdl.p_pv_curtailment[t] for t in mdl.T)

        # Battery loss cost
        f_battery_loss = mdl.objective_function_prices[5] * (
                ((1 - mdl.battery_discharging_efficiency) / mdl.battery_discharging_efficiency) *
                sum(mdl.p_battery_discharge[t] for t in mdl.T) +
                (1 - mdl.battery_charging_efficiency) *
                sum(mdl.p_battery_charge[t] for t in mdl.T)
        )

        # Hydrogen loss cost
        f_hydrogen_loss = mdl.objective_function_prices[6] * (
                ((1 - mdl.hydrogen_discharging_efficiency) / mdl.hydrogen_discharging_efficiency) *
                sum(mdl.p_hydrogen_discharge[t] for t in mdl.T) +
                (1 - mdl.hydrogen_charging_efficiency) *
                sum(mdl.p_hydrogen_charge[t] for t in mdl.T)
        )
        # Exgrid cost
        f_exgrid = (mdl.objective_function_prices[7] * sum(mdl.p_exgrid_import[t] for t in mdl.T) -
                  mdl.objective_function_prices[8] * sum(mdl.p_exgrid_export[t] for t in mdl.T))
        
        return f_load + f_diesel + f_wind_curtailed + f_pv_curtailed + f_battery_loss + f_hydrogen_loss + f_exgrid

        
    # Add the objective function to the model
    model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    return model
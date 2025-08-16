# In the name of God
# In this file the components of microgrid will be defined
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
from useful_functions import round_value

class MicroGrid:
    """
    This class represents a single micro-grid. It gets all elements and has methods for controlling MG
    """
    def __init__(self, components, name='MG1', control_strategy = 'rbc', ):

        self.name = name  # name of micro-grid
        self.components = {  # An empty dictionary for storing MG's components
            'loads': [],
            'storages': [],
            'renewables': [],
            'exgrids': [],
            'generators': [],
        }

        for component in components:  # Appending each component of MG based on it's type to specific list in
            # component's dictionary
            if isinstance(component, Load):
                self.components['loads'].append(component)
            elif isinstance(component, Storage):
                self.components['storages'].append(component)
            elif isinstance(component, Renewable):
                self.components['renewables'].append(component)
            elif isinstance(component, ExGrid):
                self.components['exgrids'].append(component)
            elif isinstance(component, Generator):
                self.components['generators'].append(component)
            else:
                raise TypeError("Unexpected component type")
        #-------------------------------------------------------------------------------------------------------------
        if not self.components['exgrids']:  # Defining microgrid_type based on connection to upstream network
            self.microgrid_type = 'Island'
        else:
            self.microgrid_type = 'Connected'
        #-------------------------------------------------------------------------------------------------------------
        self.sink_components_indices = []  # An empty list for sorting components based on their sink_cost (ascending)
        for component_type in ['generators',
                               'storages',
                               'exgrids',
                               'renewables',
                               'loads']:
            self.sink_components_indices.extend([(i, c) for i, c in enumerate(self.components[component_type])])
            # Filling the empty list by tuples of microgrid's components and their indices.
        self.sink_components_indices.sort(key=lambda item: item[1].sink_cost)  # sorting components based on sink_cost

        self.source_components_indices = []  # An empty list for sorting components based on their least source_cost
        for component_type in ['renewables',
                               'storages',
                               'exgrids',
                               'generators',
                               'loads']:
            self.source_components_indices.extend([(i, c) for i, c in enumerate(self.components[component_type])])
            # Filling the empty list by tuples of microgrid's components and their indices.
        self.source_components_indices.sort(key=lambda item: item[1].source_cost)  # sorting based on source-cost
        self.balancing_delta = [0] * len(self.components['loads'][0].p_actual) # A list that stores delta of balancing
        #------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """
        This method returns a string that represents type of microgrid, it's name and it's components
        :return: a string representation of the microgrid
        """
        return (f"  '{self.name}' is {self.microgrid_type} Microgrid.\n\t"
               f" number of loads: {len(self.components['loads'])},\n\t"
               f" number of storage systems: {len(self.components['storages'])},\n\t"
               f" number of renewable resources: {len(self.components['renewables'])},\n\t"
               f" number of external connections with other grids: {len(self.components['exgrids'])},\n\t"
               f" number of fossil fuel generators: {len(self.components['generators'])}.")

    def rbc(self, t):
        """
        This method is defined to calculate control_commands for each component of the microgrid at timestep 't',
        based on rule_based control (RBC) strategy.
        :param t: time-step
        :return: dictionary of control_commands for  components of the microgrid at timestep 't'
        """
        control_commands = {
            'loads': [],
            'storages': [],
            'renewables': [],
            'exgrids': [],
            'generators': [],
        }

        loads_sum = 0
        for load in self.components['loads']: # Sum-up all loads consumptions for time-step 't'
            loads_sum += load.p_actual[t]

        renewable_resources_sum = 0
        for renewable in self.components['renewables']: # Sum-up all renewable generations for time-step 't'
            renewable_resources_sum += renewable.p_actual[t]

        delta = renewable_resources_sum - loads_sum  # Calculating the difference between generation and consumption
        # at time step 't'

        if delta > 0 : # There is surplus energy
            surplus_energy = abs(delta)
            for index, component in self.sink_components_indices:
                surplus_energy, component_command = component.sink_planning(surplus_energy, t)
                if isinstance(component, Load):
                    control_commands['loads'].append((index, component_command))
                elif isinstance(component, Storage):
                    control_commands['storages'].append((index, component_command))
                elif isinstance(component, ExGrid):
                    control_commands['exgrids'].append((index, component_command))
                elif isinstance(component, Generator):
                    control_commands['generators'].append((index, component_command))
                elif isinstance(component, Renewable):
                    control_commands['renewables'].append((index, component_command))

        elif delta < 0:  # There is deficit of energy
            deficit_energy = abs(delta)
            for index, component in self.source_components_indices:
                deficit_energy, component_command = component.source_planning(deficit_energy, t)
                if isinstance(component, Load):
                    control_commands['loads'].append((index, component_command))
                elif isinstance(component, Storage):
                    control_commands['storages'].append((index, component_command))
                elif isinstance(component, ExGrid):
                    control_commands['exgrids'].append((index, component_command))
                elif isinstance(component, Generator):
                    control_commands['generators'].append((index, component_command))
                elif isinstance(component, Renewable):
                    control_commands['renewables'].append((index, component_command))

        else:  # There is equilibrium between renewable generation and load consumption for time-step 't'
            for index, component in self.source_components_indices:
                if isinstance(component, Load):
                    control_commands['loads'].append((index, 0.))
                elif isinstance(component, Storage):
                    control_commands['storages'].append((index, 0.))
                elif isinstance(component, ExGrid):
                    control_commands['exgrids'].append((index, 0.))
                elif isinstance(component, Generator):
                    control_commands['generators'].append((index, 0.))
                elif isinstance(component, Renewable):
                    control_commands['renewables'].append((index, 0.))

        return control_commands

    def step(self, t, control_type='rbc'):
        """
        This method gets control_type and based on that by using control methods like 'rbc' and 'mpc' calculates
        control_commands for all components of the microgrid at timestep 't'. Then using command_execution method
        that every component has, it sets those commands to internal parameters for each component and updates
        condition for that time-step 't'.
        :param t: time-step 't'
        :param control_type: The type of control strategy : {'RBC', 'MPC', 'RL'}
        :return: Microgrid
        """
        commands = {}

        if control_type == 'rbc':
            commands = self.rbc(t)
        elif control_type == 'mpc':
            pass
        elif control_type == 'rl':
            pass
        else:
            raise ValueError('Unknown control type')


        for component_type, component_commands in commands.items():
            for command in component_commands:
                component_index = command[0]
                component = self.components[component_type][component_index]
                component.command_execution(command[1],  t)

        return self

    def run_simulation(self, control_type='rbc', num_steps=None):
        for t in range(num_steps):
            self.step(t, control_type)

    def forecasting(self):
        pass

    def optimization(self, price,
                     consumption_prediction,
                     wind_prediction,
                     pv_prediction,):
        """
        This method is the central core of optimization in each time-step. The problem is going to be formulated as NLP
        problem and be solved by 'ipopt' solver .
        Args:
            price: a dictionary contains coefficients that identifies price for 1 kWh energy of each part of objective_function
            consumption_prediction: The predicted values for consumption
            wind_prediction: The predicted values for wind_production
            pv_prediction: The predicted values for pv_production

        Returns: model.

        """
        pass

    def balancing(self, t):
        """
        This method calculates delta of generated and consumed energy for time-step 't' based on actual values of
        renewables and load consumptions and commanded energy values of all components of the microgrid . Then it
        modifies commands based on 'rbc' to make balance of energy for that specific time-step.
        Args:
            t: time-step 't'

        Returns: None

        """
        # calculating energy imbalance in time-step 't':
        delta = 0
        for renewable in self.components['renewables']:
            delta += renewable.p_actual[t] - renewable.p_curtailment[t]
        for load in self.components['loads']:
            delta -= load.p_actual[t] + load.p_change[t]  # p_change for growing consumption is positive value
        for storage in self.components['storages']:
            delta -= storage.p_conv[t]  # p_conv for charging is a positive value
        for generator in self.components['generators']:
            delta += generator.p_gen[t]
        for exgrid in self.components['exgrids']:
            delta += exgrid.p_exchange[t]  # p_exchange for importing energy is a positive value

        self.balancing_delta[t] = delta  # Storing delta

        if delta > 0 : # There is surplus energy
            surplus_energy = abs(delta)
            for index, component in self.sink_components_indices:
                surplus_energy = component.sink_balance(surplus_energy, t)

        elif delta < 0:  # There is deficit of energy
            deficit_energy = abs(delta)
            for index, component in self.source_components_indices:
                deficit_energy = component.source_balance(deficit_energy, t)
        else:
            for storage in self.components['storages']:
                storage.soc_update(t)
        return None


    def results(self):
        """
        This method will save the results of simulation in form of a pandas dataframe and visualize it
        Returns: A pandas dataframe

        """
        time_stamps = self.components['loads'][0].time_stamps  # Getting date-time from load component
        timesteps = len(time_stamps)
        data = {"Time_stamp": time_stamps}  # Making date-time column
        #-----------------------------------------------Loads
        total_load = [0] * timesteps
        total_load_change = [0] * timesteps
        total_load_cost = [0] * timesteps
        #-----------------------------------------------Renewables
        total_renewable = [0] * timesteps
        total_renewable_curtailment = [0] * timesteps
        total_renewable_cost = [0] * timesteps
        #-----------------------------------------------Storages
        total_storage_convert = [0] * timesteps
        total_storage_cost = [0] * timesteps
        #-----------------------------------------------Generator
        total_fossil = [0] * timesteps
        total_fossil_cost = [0] * timesteps
        #-----------------------------------------------ExGrids
        total_exgrid_exchange = [0] * timesteps
        total_exgrid_cost = [0] * timesteps

        # Process load components if they exist
        if self.components['loads']:
            for i, load in enumerate(self.components['loads']):  # saving Load's data
                data[f"{load.name}_Actual_Consumption"] = load.p_actual
                data[f"{load.name}_Change"] = load.p_change
                data[f"{load.name}_Balance"] = load.p_balance
                
                # Add individual component costs and cumulative costs
                load_cost = load.change_cost()
                data[f"{load.name}_Cost"] = load_cost
                data[f"{load.name}_Cumulative_Cost"] = list(accumulate(load_cost))
                
                total_load = [x + y for x, y in zip(total_load, load.p_actual)]
                total_load_change = [x + y for x, y in zip(total_load_change, load.p_change)]
                total_load_cost = [x + y for x, y in zip(total_load_cost, load_cost)]
            
            # Add load component type costs only if loads exist
            data["Loads_Cost"] = total_load_cost
            data["Cumulative_Loads_Cost"] = list(accumulate(total_load_cost))

        # Process renewable components if they exist
        if self.components['renewables']:
            for i, renewable in enumerate(self.components['renewables']):  # saving Renewable's data
                data[f"{renewable.name}_Actual_Generation"] = renewable.p_actual
                data[f"{renewable.name}_Curtailment"] = renewable.p_curtailment
                data[f"{renewable.name}_Balance"] = renewable.p_balance

                # Add individual component costs and cumulative costs
                renewable_cost = renewable.curtailment_cost()
                data[f"{renewable.name}_Cost"] = renewable_cost
                data[f"{renewable.name}_Cumulative_Cost"] = list(accumulate(renewable_cost))
                
                total_renewable = [x + y for x, y in zip(total_renewable, renewable.p_actual)]
                total_renewable_curtailment = [x + y for x, y in zip(total_renewable_curtailment, renewable.p_curtailment)]
                total_renewable_cost = [x + y for x, y in zip(total_renewable_cost, renewable_cost)]
            
            # Add renewable component type costs only if renewables exist
            data["Renewables_Cost"] = total_renewable_cost
            data["Cumulative_Renewables_Cost"] = list(accumulate(total_renewable_cost))

        # Process storage components if they exist
        if self.components['storages']:
            for i, storage in enumerate(self.components['storages']):  # saving Storage's data
                data[f"{storage.name}_SOC"] = storage.soc
                data[f"{storage.name}_Convert"] = storage.p_conv
                data[f"{storage.name}_Balance"] = storage.p_balance

                # Add individual component costs and cumulative costs
                storage_cost = storage.convert_cost()
                data[f"{storage.name}_Cost"] = storage_cost
                data[f"{storage.name}_Cumulative_Cost"] = list(accumulate(storage_cost))
                
                total_storage_convert = [x + y for x, y in zip(total_storage_convert, storage.p_conv)]
                total_storage_cost = [x + y for x, y in zip(total_storage_cost, storage_cost)]
            
            # Add storage component type costs only if storages exist
            data["Storages_Cost"] = total_storage_cost
            data["Cumulative_Storages_Cost"] = list(accumulate(total_storage_cost))

        # Process generator components if they exist
        if self.components['generators']:
            for i, generator in enumerate(self.components['generators']):  # saving Generator's data
                data[f"{generator.name}_Generation"] = generator.p_gen
                data[f"{generator.name}_Balance"] = generator.p_balance

                # Add individual component costs and cumulative costs
                generator_cost = generator.generation_cost()
                data[f"{generator.name}_Cost"] = generator_cost
                data[f"{generator.name}_Cumulative_Cost"] = list(accumulate(generator_cost))
                
                total_fossil = [x + y for x, y in zip(total_fossil, generator.p_gen)]
                total_fossil_cost = [x + y for x, y in zip(total_fossil_cost, generator_cost)]
            
            # Add generator component type costs only if generators exist
            data["Generators_Cost"] = total_fossil_cost
            data["Cumulative_Generators_Cost"] = list(accumulate(total_fossil_cost))

        # Process exgrid components if they exist
        if self.components['exgrids']:
            for i, exgrid in enumerate(self.components['exgrids']):  # saving ExGrid's data
                data[f"{exgrid.name}_Exchange"] = exgrid.p_exchange
                data[f"{exgrid.name}_Balance"] = exgrid.p_balance

                # Add individual component costs and cumulative costs
                exgrid_cost = exgrid.exchange_cost()
                data[f"{exgrid.name}_Cost"] = exgrid_cost
                data[f"{exgrid.name}_Cumulative_Cost"] = list(accumulate(exgrid_cost))
                
                total_exgrid_exchange = [x + y for x, y in zip(total_exgrid_exchange, exgrid.p_exchange)]
                total_exgrid_cost = [x + y for x, y in zip(total_exgrid_cost, exgrid_cost)]
            
            # Add exgrid component type costs only if exgrids exist
            data["ExGrids_Cost"] = total_exgrid_cost
            data["Cumulative_ExGrids_Cost"] = list(accumulate(total_exgrid_cost))

        imbalance = [
            total_renewable[i] - total_renewable_curtailment[i] + total_fossil[i] +
             total_exgrid_exchange[i] - total_storage_convert[i] - total_load[i] - total_load_change[i]
            for i in range(timesteps)
        ]

        data["Energy_Imbalance"] = imbalance
        data["Balancing_delta"] = self.balancing_delta

        # Calculate total cost from all component costs
        total_cost = [0] * timesteps
        
        # Add costs from each component type that exists
        if self.components['loads']:
            total_cost = [x + y for x, y in zip(total_cost, total_load_cost)]
        if self.components['renewables']:
            total_cost = [x + y for x, y in zip(total_cost, total_renewable_cost)]
        if self.components['storages']:
            total_cost = [x + y for x, y in zip(total_cost, total_storage_cost)]
        if self.components['generators']:
            total_cost = [x + y for x, y in zip(total_cost, total_fossil_cost)]
        if self.components['exgrids']:
            total_cost = [x + y for x, y in zip(total_cost, total_exgrid_cost)]
            
        data["Total_Cost"] = total_cost
        data["Cumulative_Total_Cost"] = list(accumulate(total_cost))

        df = pd.DataFrame(data)  # converting to pandas dataframe

        return df

    def visualize(self, start_time=None, end_time=None):
        """
        This method will visualize the results of simulation in form of time-series plots.
        Args:
            start_time: The start time of the interested time period.
            end_time: The end time of the interested time period.

        Returns: None

        """
        df = self.results()
        if start_time and end_time:
            df = df[(df['Time_stamp'] >= start_time) & (df['Time_stamp'] <= end_time)]

        time_stamps = df['Time_stamp']
        df = df.drop(columns=['Time_stamp'])
        num_plots = len(df.columns)

        fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 2), sharex=True)

        for i, column in enumerate(df.columns):
            axs[i].plot(time_stamps, df[column], label=column)
            if column == 'Total_Cost' or column == 'Cumulative_Total_Cost':
                axs[i].set_ylabel('Cost ($)')
            else:
                axs[i].set_ylabel('Power (kW)')
            axs[i].set_title(column)
            axs[i].grid(True)

        plt.xlabel('Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

class Load:
    """
     This class represent the load elements in microgrid and has methods for shedding or growing loads.
    """
    def __init__(self, name,
                 time_stamps,
                 p_actual_consumption,
                 critical_load=20,
                 max_consumption_increase_rate = None,
                 max_consumption_decrease_rate = None,
                 installed_capacity = None,
                 sink_cost = 100,
                 source_cost = 100,
                 growing_price = 0.8,
                 shedding_price = 0.9):

        self.name = name  # The name of Load
        self.time_stamps = time_stamps  # Time stamps that will be used in results method.
        self.p_actual = [round_value(v) for v in list(p_actual_consumption)]  # actual consumption of load [kW]
        self.p_nom = p_actual_consumption.mean()  # Nominal power of load [kW]
        self.critical = critical_load  # The amount of load that has to be served at all cost [kW]
        self.p_change = [0] * len(p_actual_consumption) # load consumption changing command [kW] (positive = growing)
        self.p_balance = [0] * len(p_actual_consumption) # load consumption balance [kW]
        self.p_forecasted = None # Forecasted load for each time step [kW]
        self.max_consumption_increase_rate = max_consumption_increase_rate  # maximum increase rate for consumption [kW]
        self.max_consumption_decrease_rate = max_consumption_decrease_rate  # Maximum decrease rate for consumption [kW]
        self.installed_capacity = installed_capacity  # The installed capacity of load (sum of all devices nominal
        # power) [kW]
        self.sink_cost = sink_cost  # This parameter is an integer that determines the
        # order of using growing method of loads in 'rbc' control strategy.
        self.source_cost = source_cost # This parameter is an integer that determines the
        # order of using shedding method of loads in 'rbc' control strategy.
        self.growing_price = growing_price  # The price of growing load  [$/kWh]
        self.shedding_price = shedding_price  # The price of shedding load [$/kWh]

    def change_cost(self):
        """
        This method calculates the growing and shedding cost of load.
        Returns:A list containing the cost of load shedding or growing for each time-step
        """
        cost_p_change = [self.growing_price * pc if pc > 0 else -self.shedding_price * pc for pc in self.p_change]
        cost_p_balance = [self.growing_price * pb if pb > 0 else -self.shedding_price * pb for pb in self.p_balance]
        return [c1 + c2 for c1, c2 in zip(cost_p_change, cost_p_balance)]

    def sink(self, energy_2b_consumed, t):
        """
        This method will update 'p_change' parameter of load for specific time-step 't' based on the value of energy
        that needs to be consumed 'energy_2b_consumed' and internal parameters of load.
        :param energy_2b_consumed: the amount of energy that needs to be consumed [kWh]
        :param t: the specific time-step
        :return: remained energy [kWh]
        """
        remained_max_consumption_increase_rate = self.max_consumption_increase_rate - self.p_change[t]  # calculating
        # remained increase rate capability of load
        expandable_consumption = self.installed_capacity - self.p_actual[t]  # Calculating expanding capability of the
        # load
        a = min(expandable_consumption, remained_max_consumption_increase_rate)  # Load can increase its consumption
        # by this amount.
        if energy_2b_consumed >= a:  # If energy that needs to be consumed is greater than the 'a' then p_change for
            # that specific time-step will increase by the amount of 'a', and then remained energy that this load could
            # not consume is calculated and returned.
            self.p_change[t] += a
            remained_energy = energy_2b_consumed - a
        else:  # If energy that needs to be consumed is less than the 'a', then whole energy will be consumed by this
            # load and p_change for that specific time-step will increase by the amount of 'energy_2b_consumed'.
            # and there will be no remained energy.
            self.p_change[t] += energy_2b_consumed
            remained_energy = 0
        return remained_energy

    def sink_balance(self, energy_2b_consumed, t):
        """
        This method will update 'p_change' parameter of load for specific time-step 't' based on the value of energy
        that needs to be consumed 'energy_2b_consumed' and internal parameters of load.
        :param energy_2b_consumed: the amount of energy that needs to be consumed [kWh]
        :param t: the specific time-step
        :return: remained energy [kWh]
        """
        remained_max_consumption_increase_rate = self.max_consumption_increase_rate - self.p_change[t]  # calculating
        # remained increase rate capability of load
        expandable_consumption = self.installed_capacity - self.p_actual[t]  # Calculating expanding capability of the
        # load
        a = min(expandable_consumption, remained_max_consumption_increase_rate)  # Load can increase its consumption
        # by this amount.
        if energy_2b_consumed >= a:  # If energy that needs to be consumed is greater than the 'a' then p_change for
            # that specific time-step will increase by the amount of 'a', and then remained energy that this load could
            # not consume is calculated and returned.
            self.p_change[t] += a
            self.p_balance[t] = a
            remained_energy = energy_2b_consumed - a
        else:  # If energy that needs to be consumed is less than the 'a', then whole energy will be consumed by this
            # load and p_change for that specific time-step will increase by the amount of 'energy_2b_consumed'.
            # and there will be no remained energy.
            self.p_change[t] += energy_2b_consumed
            self.p_balance[t] = energy_2b_consumed
            remained_energy = 0
        return remained_energy

    def sink_planning(self, energy_2b_consumed, t):
        """
        This method will calculate change command of load for specific time-step 't' based on the value of energy
        that needs to be consumed 'energy_2b_consumed' and internal parameters of load.
        :param energy_2b_consumed: The amount of energy that needs to be consumed [kWh]
        :param t: the specific time-step
        :return: remained energy and change_command
        """
        change_command = self.p_change[t]  # initial value of change command is equal to p_change[t]
        remained_max_consumption_increase_rate = self.max_consumption_increase_rate - change_command  # calculating
        # remained increase rate capability of load
        expandable_consumption = self.installed_capacity - self.p_actual[t]  # Calculating expanding capability of the
        # load
        a = min(expandable_consumption, remained_max_consumption_increase_rate)  # Load can increase its consumption
        # by this amount.
        if energy_2b_consumed >= a:  # If energy that needs to be consumed is greater than the 'a' then change command
            # for that specific time-step will increase by the amount of 'a', and then remained energy that this load
            # could not consume is calculated and returned.
            change_command += a
            remained_energy = energy_2b_consumed - a
        else:  # If energy that needs to be consumed is less than the 'a', then whole energy will be consumed by this
            # load and change command for that specific time-step will increase by the amount of 'energy_2b_consumed'.
            # and there will be no remained energy.
            change_command += energy_2b_consumed
            remained_energy = 0
        return remained_energy, change_command

    def source(self, energy_2b_shed, t):
        """
        This method will update 'p_change' parameter of load for specific time-step 't' based on the amount of deficit
        energy that needs to be shed 'energy_2b_shed' and internal parameters of load.
        :param energy_2b_shed: The amount of deficit energy that needs to be shed [kWh]
        :param t: The specific time-step
        :return: remained_energy [kWh]
        """
        remained_max_consumption_decrease_rate = self.max_consumption_decrease_rate + self.p_change[t]  # Calculating
        # remained decrease rate of consumption capability of load.
        a = min(self.p_actual[t], remained_max_consumption_decrease_rate)  # Load can decrease its consumption by this
        # amount
        if energy_2b_shed >= a:  # If energy that needs to be shed is greater than the 'a' then p_change for
            # that specific time-step will decrease by the amount of 'a', and then remained energy that this load could
            # not shed is calculated and returned.
            self.p_change[t] -= a
            remained_energy = energy_2b_shed - a
        else:  # If energy that needs to be shed is less than the 'a', then whole energy will be shed by this
            # load and p_change for that specific time-step will decrease by the amount of 'energy_2b_shed'.
            # and there will be no remained energy.
            self.p_change[t] -= energy_2b_shed
            remained_energy = 0
        return remained_energy

    def source_balance(self, energy_2b_shed, t):
        """
        This method will update 'p_change' parameter of load for specific time-step 't' based on the amount of deficit
        energy that needs to be shed 'energy_2b_shed' and internal parameters of load.
        :param energy_2b_shed: The amount of deficit energy that needs to be shed [kWh]
        :param t: The specific time-step
        :return: remained_energy [kWh]
        """
        remained_max_consumption_decrease_rate = self.max_consumption_decrease_rate + self.p_change[t]  # Calculating
        # remained decrease rate of consumption capability of load.
        a = min(self.p_actual[t], remained_max_consumption_decrease_rate)  # Load can decrease its consumption by this
        # amount
        if energy_2b_shed >= a:  # If energy that needs to be shed is greater than the 'a' then p_change for
            # that specific time-step will decrease by the amount of 'a', and then remained energy that this load could
            # not shed is calculated and returned.
            self.p_change[t] -= a
            self.p_balance[t] = a
            remained_energy = energy_2b_shed - a
        else:  # If energy that needs to be shed is less than the 'a', then whole energy will be shed by this
            # load and p_change for that specific time-step will decrease by the amount of 'energy_2b_shed'.
            # and there will be no remained energy.
            self.p_change[t] -= energy_2b_shed
            self.p_balance[t] = energy_2b_shed
            remained_energy = 0
        return remained_energy


    def source_planning(self, energy_2b_shed, t):
        """
        This method will calculate change_command of load for specific time-step 't' based on the amount of deficit
        energy that needs to be shed 'energy_2b_shed' and internal parameters of load.
        :param energy_2b_shed: The amount of deficit energy that needs to be shed [kWh]
        :param t: The specific time-step
        :return: remained_energy and change_command
        """
        change_command = self.p_change[t]  # initial value of change_command is equal to p_change[t]
        remained_max_consumption_decrease_rate = self.max_consumption_decrease_rate + change_command  # Calculating
        # remained decrease rate of consumption capability of load.
        a = min(self.p_actual[t], remained_max_consumption_decrease_rate)  # Load can decrease its consumption by this
        # amount
        if energy_2b_shed >= a:  # If energy that needs to be shed is greater than the 'a' then change_command
            # will decrease by the amount of 'a', and then remained energy that this load could not shed is calculated
            # and returned.
            change_command -= a
            remained_energy = energy_2b_shed - a
        else:  # If energy that needs to be shed is less than the 'a', then whole energy will be shed by this
            # load and change_command will decrease by the amount of 'energy_2b_shed' and there will be no
            # remained energy.
            change_command -= energy_2b_shed
            remained_energy = 0
        return remained_energy, change_command

    def command_execution(self, change_command, t):
        """
        This method will get change_command that calculated in another method. and will set in to p_change[t]
        Args:
            change_command: The calculated change command of load [kW]
            t: time-step

        Returns: None

        """
        self.p_change[t] = change_command

class ExGrid:
    """
    This class represent the External Grid elements in microgrid
    """
    def __init__(self, name,
                 p_max_import,
                 p_max_export,
                 sink_cost=50,
                 source_cost=50,
                 length=None,
                 import_price = 0.06,
                 export_price = 0.06,):

        self.name = name  # The name of External Grid
        self.p_max_import = p_max_import  # This is maximum power that can be imported from ExGrid [kW]
        self.p_max_export = p_max_export  # This is maximum power that can be exported to ExGrid [kW]
        self.p_exchange = [0] * length  # import power ordered by EMS (positive = import)
        self.p_balance = [0] * length  # import power balance
        self.sink_cost = sink_cost
        self.source_cost = source_cost
        self.import_price = import_price  # The price of importing energy from external grid [$/kWh]
        self.export_price = export_price  # The price of exporting energy to external grid [$/kWh]

    def exchange_cost(self):
        """
        This method calculates the cost of import/export energy from/to External-grid
        Returns:A list that containing cost of importing and exporting for each time-step
        """
        cost_p_exchange = [self.import_price * p_ex if p_ex > 0 else self.export_price * p_ex for  p_ex in self.p_exchange]
        cost_p_balance = [self.import_price * p_bal if p_bal > 0 else self.export_price * p_bal for  p_bal in self.p_balance]
        return [c1 + c2 for c1, c2 in zip(cost_p_exchange, cost_p_balance)]

    def source(self, shortage, t):
        """
        This method imports shortage energy that microgrid needs from external grid
        :param t: time-step 't'
        :param shortage: the amount of energy that is needed [kWh]
        :return: remained_needed_energy
        """
        remained_max_import = self.p_max_import - self.p_exchange[t]
        if shortage <= remained_max_import:  # Shortage energy is less than maximum import rate of external grid
            self.p_exchange[t] += shortage  # Import whole needed energy from external grid
            remained_needed_energy = 0  # There will be no more shortage
        else:  # Shortage energy is greater than maximum import rate of external grid
            self.p_exchange[t] += remained_max_import # Import from external grid based on maximum import rate
            remained_needed_energy = shortage - remained_max_import  # Calculate the remained needed energy
        return remained_needed_energy

    def source_balance(self, shortage, t):
        """
        This method imports shortage energy that microgrid needs from external grid
        :param t: time-step 't'
        :param shortage: the amount of energy that is needed [kWh]
        :return: remained_needed_energy
        """
        remained_max_import = self.p_max_import - self.p_exchange[t]
        if shortage <= remained_max_import:  # Shortage energy is less than maximum import rate of external grid
            self.p_exchange[t] += shortage  # Import whole needed energy from external grid
            self.p_balance[t] = shortage
            remained_needed_energy = 0  # There will be no more shortage
        else:  # Shortage energy is greater than maximum import rate of external grid
            self.p_exchange[t] += remained_max_import # Import from external grid based on maximum import rate
            self.p_balance[t] = remained_max_import
            remained_needed_energy = shortage - remained_max_import  # Calculate the remained needed energy
        return remained_needed_energy

    def source_planning(self, shortage, t):
        """
        This method calculates exchange_command for an External_grid based on the amount of energy needed and internal
        parameters of that External_grid
        :param shortage: The amount of energy that is needed [kWh]
        :param t: time-step 't'
        :return: remained_needed_energy and exchange_command
        """
        exchange_command = self.p_exchange[t]
        remained_max_import = self.p_max_import - exchange_command
        if shortage <= remained_max_import:  # Shortage energy is less than maximum import rate of external grid
            exchange_command += shortage  # Import whole needed energy from external grid
            remained_needed_energy = 0  # There will be no more shortage
        else:  # Shortage energy is greater than maximum import rate of external grid
            exchange_command += remained_max_import # Import from external grid based on maximum import rate
            remained_needed_energy = shortage - remained_max_import  # Calculate the remained needed energy
        return remained_needed_energy, exchange_command

    def sink(self, surplus, t):
        """
        This method exports surplus energy from microgrid to external grid
        :param t: time-step 't'
        :param surplus: the amount of extra energy [kWh]
        :return: remained_extra
        """
        remained_max_export = self.p_max_export + self.p_exchange[t]
        if surplus <= remained_max_export:  # extra energy is less than maximum export rate of external grid
            self.p_exchange[t] -= surplus # Export whole extra energy to external grid
            remained_extra_energy = 0  # There will be no more extra
        else:  # Surplus energy is greater than maximum export rate of external grid
            self.p_exchange[t] -= remained_max_export  # Export to external grid based on maximum export rate
            remained_extra_energy = surplus - remained_max_export  # Calculate the remained extra energy
        return remained_extra_energy

    def sink_balance(self, surplus, t):
        """
        This method exports surplus energy from microgrid to external grid
        :param t: time-step 't'
        :param surplus: the amount of extra energy [kWh]
        :return: remained_extra
        """
        remained_max_export = self.p_max_export + self.p_exchange[t]
        if surplus <= remained_max_export:  # extra energy is less than maximum export rate of external grid
            self.p_exchange[t] -= surplus # Export whole extra energy to external grid
            self.p_balance[t] = surplus
            remained_extra_energy = 0  # There will be no more extra
        else:  # Surplus energy is greater than maximum export rate of external grid
            self.p_exchange[t] -= remained_max_export  # Export to external grid based on maximum export rate
            self.p_balance[t] = remained_max_export
            remained_extra_energy = surplus - remained_max_export  # Calculate the remained extra energy
        return remained_extra_energy

    def sink_planning(self, surplus, t):
        """
        This method calculates exchange_command for an External_grid based on the amount of surplus energy and internal
        parameters of that External_grid
        :param surplus: The amount of extra energy [kWh]
        :param t: time-step 't'
        :return: remained_extra_energy and exchange_command
        """
        exchange_command = self.p_exchange[t]
        remained_max_export = self.p_max_export + exchange_command
        if surplus <= remained_max_export:  # extra energy is less than maximum export rate of external grid
            exchange_command -= surplus # Export whole extra energy to external grid
            remained_extra_energy = 0  # There will be no more extra
        else:  # Surplus energy is greater than maximum export rate of external grid
            exchange_command -= remained_max_export  # Export to external grid based on maximum export rate
            remained_extra_energy = surplus - remained_max_export  # Calculate the remained extra energy
        return remained_extra_energy, exchange_command

    def command_execution(self, exchange_command, t):
        """
        This method will get exchange_command that calculated in another method. and will set in to p_exchange[t]
        Args:
            exchange_command: The calculated exchange command of load [kW]
            t: time-step

        Returns: None

        """
        self.p_exchange[t] = exchange_command

class Generator:
    """
    This class represent the controllable fossil fuel generator elements in microgrid
    """
    def __init__(self, name,
                 p_nom,
                 length,
                 sink_cost=10,
                 source_cost=90,
                 generation_price = 0.5):

        self.name = name
        self.p_nom = p_nom  # Nominal power of the generator [kW]
        self.p_gen = [0] * length  # ordered generation of generator by EMS for each time step [kW]
        self.p_balance = [0] * length  # generation balance
        self.sink_cost = sink_cost
        self.source_cost = source_cost
        self.generation_price = generation_price  #  The price of generating 1 kWh energy [$/kWh] by this generator

    def generation_cost(self):
        """
        This method calculates the generation cost of this generator
        Returns: A list of generation cost for each time-step
        """
        cost_p_gen = [self.generation_price * pg for pg in self.p_gen]
        cost_p_balance = [self.generation_price * pb for pb in self.p_balance]
        return [c1 + c2 for c1, c2 in zip(cost_p_gen, cost_p_balance)]

    def sink(self, surplus_energy, t):
        """
        This method modifies p_gen for time-step 't' based on the amount of surplus_energy and internal parameters
        of that Generator
        :param surplus_energy: The amount of surplus energy [kWh]
        :param t: time-step 't'
        :return: remained_surplus_energy,
        """
        if surplus_energy <= self.p_gen[t]:
            self.p_gen[t] -= surplus_energy
            remained_surplus_energy = 0
        else:
            remained_surplus_energy = surplus_energy - self.p_gen[t]
            self.p_gen[t] = 0
        return remained_surplus_energy

    def sink_balance(self, surplus_energy, t):
        """
        This method modifies p_gen for time-step 't' based on the amount of surplus_energy and internal parameters
        of that Generator
        :param surplus_energy: The amount of surplus energy [kWh]
        :param t: time-step 't'
        :return: remained_surplus_energy,
        """
        if surplus_energy <= self.p_gen[t]:
            self.p_gen[t] -= surplus_energy
            self.p_balance[t] = surplus_energy
            remained_surplus_energy = 0
        else:
            remained_surplus_energy = surplus_energy - self.p_gen[t]
            self.p_balance[t] = self.p_gen[t]
            self.p_gen[t] = 0
        return remained_surplus_energy

    def sink_planning(self, surplus_energy, t):
        """
        This method calculates generate_command for time-step 't' based on the amount of surplus_energy and internal
         parameters of that Generator
        :param surplus_energy: The amount of surplus energy [kWh]
        :param t: time-step 't'
        :return: remained_surplus_energy, generate_command,
        """
        generate_command = self.p_gen[t]
        if surplus_energy <= generate_command:
            generate_command -= surplus_energy
            remained_surplus_energy = 0
        else:
            remained_surplus_energy = surplus_energy - generate_command
            generate_command = 0
        return remained_surplus_energy, generate_command

    def source(self, deficit_energy, t):
        """
        This method modifies p_gen for time-step 't' based on the amount of deficit_energy and internal parameters
        of that Generator
        :param deficit_energy: The amount of deficit energy [kWh]
        :param t: time-step 't'
        :return: remained_deficit_energy,
        """
        a = self.p_nom - self.p_gen[t]
        if deficit_energy <= a:
            self.p_gen[t] += deficit_energy
            remained_deficit_energy = 0
        else:
            remained_deficit_energy = deficit_energy - a
            self.p_gen[t] += a
        return remained_deficit_energy

    def source_balance(self, deficit_energy, t):
        """
        This method modifies p_gen for time-step 't' based on the amount of deficit_energy and internal parameters
        of that Generator
        :param deficit_energy: The amount of deficit energy [kWh]
        :param t: time-step 't'
        :return: remained_deficit_energy,
        """
        a = self.p_nom - self.p_gen[t]
        if deficit_energy <= a:
            self.p_gen[t] += deficit_energy
            self.p_balance[t] = deficit_energy
            remained_deficit_energy = 0
        else:
            remained_deficit_energy = deficit_energy - a
            self.p_gen[t] += a
            self.p_balance[t] = a
        return remained_deficit_energy

    def source_planning(self, deficit_energy, t):
        """
        This method calculates generate_command for time-step 't' based on the amount of deficit_energy and internal
        parameters of that Generator
        :param deficit_energy: The amount of deficit energy [kWh]
        :param t: time-step 't'
        :return: remained_deficit_energy, generate_command,
        """
        generate_command = self.p_gen[t]
        a = self.p_nom - self.p_gen[t]
        if deficit_energy <= a:
            generate_command += deficit_energy
            remained_deficit_energy = 0
        else:
            remained_deficit_energy = deficit_energy - a
            generate_command += a
        return remained_deficit_energy, generate_command

    def command_execution(self, generate_command, t):
        """
        This method gets calculated generate_command and set it on p_gen for time-step 't'
        :param generate_command: The calculated command for generation of Generator [kWh]
        :param t: time-step 't'
        :return: None
        """
        if generate_command < 0:
            raise ValueError("Generation command must be greater than or equal to 0")
        else:
            self.p_gen[t] = generate_command
        return None

class Renewable:
    """
    This class represent the renewable generators in microgrid, wind and photovoltaic
    """
    def __init__(self, name,
                 p_actual_generation,
                 p_nom,
                 sink_cost=80,
                 source_cost=20,
                 curtailment_price = 0.1):

        self.name = name
        self.p_actual = [round_value(v) for v in list(p_actual_generation)]  # Actual generated power of renewable [kW]
        self.p_nom = p_nom  # Nominal power of renewable generator [kW]
        self.p_curtailment = [0] * len(p_actual_generation)  # Instantaneous curtailment of renewable generator [kW]
        self.p_balance = [0] * len(p_actual_generation)  # curtailment balance
        self.p_forecasted = None # Forecasted values of generated power of renewable [kW]
        self.sink_cost = sink_cost
        self.source_cost = source_cost
        self.curtailment_price = curtailment_price  # The price of curtailing the energy of Renewable [$/kWh]

    def curtailment_cost(self):
        """
        This method calculates the curtailment cost
        Returns: A list of curtailment costs for each time-step
        """
        cost_p_curtailment = [self.curtailment_price * pc  for pc in self.p_curtailment]
        cost_p_balance = [self.curtailment_price * pb for pb in self.p_balance]
        return [c1 + c2 for c1, c2 in zip(cost_p_curtailment, cost_p_balance)]

    def sink(self, p_extra, t):
        """
        This method gets extra energy that needs to be curtailed and does the curtailing
        :param t: current time step 't'
        :param p_extra: the amount of energy that is extra and need to be curtailed
        :return: p_remained_extra: the amount of energy that remained after curtailing
        """
        remained_curtailment_capacity = self.p_actual[t] - self.p_curtailment[t]
        if p_extra <= remained_curtailment_capacity:  # Extra energy is less than or equal to renewable generation
            self.p_curtailment[t] += p_extra  # Curtail the whole extra energy
            p_remained_extra = 0  # There will be no more energy to be curtailed
        else:  # Extra energy is greater than renewable generation
            self.p_curtailment[t] += remained_curtailment_capacity  # Curtail the whole generation of renewable unit
            p_remained_extra = p_extra - remained_curtailment_capacity  # Calculate remained energy that needs to bu curtailed
        return p_remained_extra

    def sink_balance(self, p_extra, t):
        """
        This method gets extra energy that needs to be curtailed and does the curtailing
        :param t: current time step 't'
        :param p_extra: the amount of energy that is extra and need to be curtailed
        :return: p_remained_extra: the amount of energy that remained after curtailing
        """
        remained_curtailment_capacity = self.p_actual[t] - self.p_curtailment[t]
        if p_extra <= remained_curtailment_capacity:  # Extra energy is less than or equal to renewable generation
            self.p_curtailment[t] += p_extra  # Curtail the whole extra energy
            self.p_balance[t] = p_extra
            p_remained_extra = 0  # There will be no more energy to be curtailed
        else:  # Extra energy is greater than renewable generation
            self.p_curtailment[t] += remained_curtailment_capacity  # Curtail the whole generation of renewable unit
            self.p_balance[t] = remained_curtailment_capacity
            p_remained_extra = p_extra - remained_curtailment_capacity  # Calculate remained energy that needs to bu curtailed
        return p_remained_extra

    def sink_planning(self, energy_2b_curtailed, t):
        """
        This method calculates curtail_command based on energy_2b_curtailed and internal parameters of that Renewable
        :param energy_2b_curtailed: The amount of energy that needs to be curtailed [kWh]
        :param t: time-step 't'
        :return: remained_energy, curtail_command,
        """
        curtail_command = self.p_curtailment[t]
        remained_curtailment_capacity = self.p_actual[t] - self.p_curtailment[t]

        if energy_2b_curtailed <= remained_curtailment_capacity:
            curtail_command += energy_2b_curtailed
            remained_energy = 0
        else:
            curtail_command += remained_curtailment_capacity
            remained_energy = energy_2b_curtailed - remained_curtailment_capacity
        return remained_energy, curtail_command

    def source(self, deficit_energy, t):
        """
        This method will try to handle deficit_energy by decreasing the amount of p_curtailment of Renewable for
        time_step 't'
        :param deficit_energy: The amount of deficit energy [kWh]
        :param t: time-step 't'
        :return: remained_deficit_energy,
        """
        if deficit_energy <= self.p_curtailment[t]:
            self.p_curtailment[t] -= deficit_energy
            remained_deficit_energy = 0
        else:
            remained_deficit_energy = deficit_energy - self.p_curtailment[t]
            self.p_curtailment[t] = 0

        return remained_deficit_energy

    def source_balance(self, deficit_energy, t):
        """
        This method will try to handle deficit_energy by decreasing the amount of p_curtailment of Renewable for
        time_step 't'
        :param deficit_energy: The amount of deficit energy [kWh]
        :param t: time-step 't'
        :return: remained_deficit_energy,
        """
        if deficit_energy <= self.p_curtailment[t]:
            self.p_curtailment[t] -= deficit_energy
            self.p_balance[t] = deficit_energy
            remained_deficit_energy = 0
        else:
            remained_deficit_energy = deficit_energy - self.p_curtailment[t]
            self.p_balance[t] = self.p_curtailment[t]
            self.p_curtailment[t] = 0

        return remained_deficit_energy

    def source_planning(self, deficit_energy, t):
        """
        This method calculates the curtail_command of the Renewable based on deficit_energy and internal parameters of
        that Renewable
        :param deficit_energy: The amount of deficit energy [kWh]
        :param t: time-step 't'
        :return: remained_deficit_energy, curtail_command,
        """

        curtail_command = self.p_curtailment[t]
        if deficit_energy <= curtail_command:
            curtail_command -= deficit_energy
            remained_deficit_energy = 0
        else:
            remained_deficit_energy = deficit_energy - curtail_command
            curtail_command = 0

        return remained_deficit_energy, curtail_command


    def command_execution(self, curtail_command, t):
        """
        This method gets calculated curtail_command and sets p_curtailment of Renewable on that amount
        :param curtail_command: The calculated curtail_command of Renewable [kWh]
        :param t: time-step 't'
        :return: None
        """
        if curtail_command < 0:
            raise ValueError("curtail_command must be greater than or equal to 0")
        else:
            self.p_curtailment[t] = curtail_command
        return None

class Storage:
    """
    This class represent the storage elements in microgrid
    """
    def __init__(self, name,
                 capacity,
                 max_charge_rate,
                 max_discharge_rate,
                 charging_efficiency,
                 discharging_efficiency,
                 initial_soc,
                 step_numbers,
                 sink_cost=30,
                 source_cost=30,
                 energy_loss_price = 0.075):

        self.name = name  # Name of Storage
        self.capacity = capacity  # Capacity of storage element [kWh]
        self.max_charge_rate = max_charge_rate  # Maximum charging rate of storage element [kW]
        self.max_discharge_rate = max_discharge_rate  # Maximum discharging rate of storage element [kW]
        self.charging_efficiency = charging_efficiency  # Charging efficiency [%]
        self.discharging_efficiency = discharging_efficiency  # Discharging efficiency [%]
        self.p_conv = [0] * step_numbers  # convert power command by EMS for each time step [kW] (positive = charge)
        self.p_balance = [0] * step_numbers  # convert power balance
        self.soc = [0] * step_numbers  # State of charge of storage for each time step [kW]
        self.soc[0] = initial_soc  # Setting initial soc of storage
        self.sink_cost = sink_cost
        self.source_cost = source_cost
        self.energy_loss_price = energy_loss_price  # The price of energy loss for storage device [$/kWh]

    def convert_cost(self):
        """
        This method calculates the energy loss cost for storage system for each time-step
        Returns: A list containing the energy loss cost for each time-step

        """
        cost_p_conv = [
            self.energy_loss_price * pc * (1 - self.charging_efficiency) if pc > 0 
            else -self.energy_loss_price * pc * ((1 - self.discharging_efficiency) / self.discharging_efficiency) 
            for pc in  self.p_conv
        ]
        cost_p_balance = [
            self.energy_loss_price * pb * (1 - self.charging_efficiency) if pb > 0 
            else -self.energy_loss_price * pb * ((1 - self.discharging_efficiency) / self.discharging_efficiency) 
            for pb in  self.p_balance
        ]
        return [c1 + c2 for c1, c2 in zip(cost_p_conv, cost_p_balance)]

    def sink(self, energy_2b_charged, t):
        """
        This method tries to charge the Storage (set p_conv variable) for time-step 't' based on energy_2b_charged
        and internal parameters of storage.
        :param energy_2b_charged: The amount of energy that is surplus [kWh]
        :param t: time-step 't'
        :return: remained_energy,
        """
        remained_max_charge_rate = self.max_charge_rate - self.p_conv[t]
        free_capacity = (self.capacity - self.soc[t])/self.charging_efficiency
        a = min(free_capacity, remained_max_charge_rate)
        if energy_2b_charged >= a:
            self.p_conv[t] += a
            remained_energy = energy_2b_charged - a
        else:
            self.p_conv[t] += energy_2b_charged
            remained_energy = 0
        self.soc_update(t)  # Updating SoC based on new value of p_conv for time-step 't'
        return remained_energy

    def sink_balance(self, energy_2b_charged, t):
        """
        This method tries to charge the Storage (set p_conv variable) for time-step 't' based on energy_2b_charged
        and internal parameters of storage.
        :param energy_2b_charged: The amount of energy that is surplus [kWh]
        :param t: time-step 't'
        :return: remained_energy,
        """
        remained_max_charge_rate = self.max_charge_rate - self.p_conv[t]
        free_capacity = (self.capacity - self.soc[t])/self.charging_efficiency
        a = min(free_capacity, remained_max_charge_rate)
        if energy_2b_charged >= a:
            self.p_conv[t] += a
            self.p_balance[t] = a
            remained_energy = energy_2b_charged - a
        else:
            self.p_conv[t] += energy_2b_charged
            self.p_balance[t] = energy_2b_charged
            remained_energy = 0
        self.soc_update(t)  # Updating SoC based on new value of p_conv for time-step 't'
        return remained_energy

    def sink_planning(self, energy_2b_charged, t):
        """
        This method will try to calculate convert_command for Storage based on energy_2b_charged and internal parameters
        of it.
        :param energy_2b_charged:The amount of energy that is surplus [kWh]
        :param t: time-step 't'
        :return: remained_energy, charge_command,
        """
        convert_command = self.p_conv[t]
        remained_max_charge_rate = self.max_charge_rate - convert_command
        free_capacity = (self.capacity - self.soc[t])/self.charging_efficiency
        a = min(free_capacity, remained_max_charge_rate)
        if energy_2b_charged >= a:
            convert_command += a
            remained_energy = energy_2b_charged - a
        else:
            convert_command += energy_2b_charged
            remained_energy = 0
        return remained_energy, convert_command

    def source(self, energy_2b_discharged, t):
        """
        This method tries to discharge the Storage (set p_conv variable) for time-step 't' based on
         energy_2b_discharged and internal parameters of storage.
        :param energy_2b_discharged: The amount of energy that is deficit and needs to be provided by Storage [kWh]
        :param t: time-step 't'
        :return: remained_energy,
        """
        remained_max_discharge_rate = self.max_discharge_rate + self.p_conv[t]
        a = min(self.soc[t] * self.discharging_efficiency, remained_max_discharge_rate)
        if energy_2b_discharged >= a:
            self.p_conv[t] -= a
            remained_energy = energy_2b_discharged - a
        else:
            self.p_conv[t] -= energy_2b_discharged
            remained_energy = 0
        self.soc_update(t)  # Updating SoC based on new value of p_conv for time-step 't'
        return remained_energy

    def source_balance(self, energy_2b_discharged, t):
        """
        This method tries to discharge the Storage (set p_conv variable) for time-step 't' based on
         energy_2b_discharged and internal parameters of storage.
        :param energy_2b_discharged: The amount of energy that is deficit and needs to be provided by Storage [kWh]
        :param t: time-step 't'
        :return: remained_energy,
        """
        remained_max_discharge_rate = self.max_discharge_rate + self.p_conv[t]
        a = min(self.soc[t] * self.discharging_efficiency, remained_max_discharge_rate)
        if energy_2b_discharged >= a:
            self.p_conv[t] -= a
            self.p_balance[t] = a
            remained_energy = energy_2b_discharged - a
        else:
            self.p_conv[t] -= energy_2b_discharged
            self.p_balance[t] = energy_2b_discharged
            remained_energy = 0
        self.soc_update(t)  # Updating SoC based on new value of p_conv for time-step 't'
        return remained_energy

    def source_planning(self, energy_2b_discharged, t):
        """
        This method will try to calculate convert_command for Storage based on energy_2b_discharged and
         internal parameters of it.
        :param energy_2b_discharged: The amount of energy that is deficit and needs to be provided by Storage [kWh]
        :param t: time-step 't'
        :return: remained_energy, convert_command,
        """
        convert_command = self.p_conv[t]
        remained_max_discharge_rate = self.max_discharge_rate + convert_command
        a = min(self.soc[t] * self.discharging_efficiency, remained_max_discharge_rate)
        if energy_2b_discharged >= a:
            convert_command -= a
            remained_energy = energy_2b_discharged - a
        else:
            convert_command -= energy_2b_discharged
            remained_energy = 0
        return remained_energy, convert_command

    def soc_update(self, t):
        """
        This method will update the Storage soc variable for time-step 't+1' based on p_conv parameter of it
        for time-step 't'.
        :param t: time-step 't'
        :return: imbalance
        """
        if self.p_conv[t] > 0:  # charging
            self.soc[t+1] = self.soc[t] + self.charging_efficiency * self.p_conv[t]
        else:  # discharging
            self.soc[t+1] = self.soc[t]  + self.p_conv[t] / self.discharging_efficiency

        if self.soc[t+1] < 0 :  # soc of Storage can't be negative
            imbalance =  self.soc[t+1]
            self.soc[t+1] = 0
        elif self.soc[t+1] > self.capacity :  # soc of Storage can't be greater than capacity of Storage
            imbalance = self.soc[t+1] - self.capacity
            self.soc[t+1] = self.capacity
        else:
            imbalance = 0
        return imbalance


    def command_execution(self, convert_command, t):
        """
        This method will get convert_command that calculated in another method. and will set in to p_conv[t] and then
        It will update the SoC of Storage by that p_conv[t]
        Args:
            convert_command: The calculated exchange command of load [kW]
            t: time-step

        Returns: None

        """
        self.p_conv[t] = convert_command
        self.soc_update(t)
        return None
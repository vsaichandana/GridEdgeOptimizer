# -----------------------------------------------------------------------------
# GridEdgeOptimizer - A Comprehensive Grid Optimization Solution
# -----------------------------------------------------------------------------
#
# Version: 1.0
# Author: Sai Chandana Vallam Kondu, Iowa State University
# Copyright (c) 2023-2024
#
# This software is proprietary and confidential. Unauthorized copying,
# distribution, or use is strictly prohibited. All rights reserved by the author.
#
# Any use of this software requires explicit permission from the author and
# Iowa State University.
#
# This software is developed as a submodule to the next generation of the ISU-DERMS platform,
# which is supported by the U.S. Department of Energy (DOE) Cybersecurity, Energy Security,
# and Emergency Response (CESER) Award under Cooperative Agreement CR000016.


from gurobipy import Model, GRB
import numpy as np
import pandas as pd

import base64
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logging
from .utils import setup_logging


class GridEdgeOptimizer:
    """Main solar and storage class using Gurobi for grid-edge optimization"""

    def __init__(self, config: dict):
        """Initialize with JSON configuration data."""
        self.config = config
        self.logger = setup_logging()

        self.hours_per_day = self.config["optimization"]["hours_per_day"]
        self.prices = self.config["optimization"]["prices"]
        self.solar_generation = self.config["optimization"]["solar_generation"]

        self.battery_soc_min = self.config["battery"]["soc_min"]
        self.battery_soc_max = self.config["battery"]["soc_max"]
        self.battery_capacity = self.config["battery"]["capacity"]
        self.power_rating = self.config["battery"]["power_rating"]
        self.eta_charge = self.config["battery"]["eta_charge"]
        self.eta_discharge = self.config["battery"]["eta_discharge"]
        self.grid_connection_capacity = self.config["solar"]["grid_connection_capacity"]

        self.logger.info("Initializing model...")
        self.setup_model()

    def setup_model(self):
        """Setup Gurobi model, variables, and constraints."""
        self.model = Model("GridEdgeOptimizer")

        # Define variables
        self.create_variables()

        # Add constraints
        self.add_constraints()

        # Define objective
        self.create_objective()

        # Set Gurobi solver parameters
        self.model.setParam("Threads", self.config["gurobi"]["threads"])

    def create_variables(self):
        """Create variables for the optimization model."""
        self.battery_power_charge = self.model.addVars(
            self.hours_per_day, lb=0, ub=self.power_rating, name="battery_charge"
        )
        self.power_discharge = self.model.addVars(
            self.hours_per_day, lb=0, ub=self.power_rating, name="battery_discharge"
        )
        self.battery_soc = self.model.addVars(
            self.hours_per_day + 1,
            lb=self.battery_soc_min * self.battery_capacity,
            ub=self.battery_soc_max * self.battery_capacity,
            name="battery_soc",
        )
        self.charging = self.model.addVars(
            self.hours_per_day, vtype=GRB.BINARY, name="charging"
        )
        self.power_solar_to_battery = self.model.addVars(
            self.hours_per_day, lb=0, name="solar_to_battery"
        )

    def create_objective(self):
        """Define the objective function to maximize profit."""
        self.logger.info("Setting up objective function...")
        self.model.setObjective(
            sum(
                self.prices[i]
                * (self.power_discharge[i] - self.battery_power_charge[i])
                for i in range(self.hours_per_day)
            )
            + sum(
                self.prices[i]
                * (self.solar_generation[i] - self.power_solar_to_battery[i])
                for i in range(self.hours_per_day)
            ),
            GRB.MAXIMIZE,
        )

    def add_constraints(self):
        """Add operational constraints to the model."""
        self.logger.info("Adding constraints...")

        # Initial state of charge
        self.model.addConstr(self.battery_soc[0] == 0, "Initial SOC")

        for i in range(self.hours_per_day):
            # Charging and discharging constraints
            self.model.addConstr(
                self.battery_power_charge[i] <= self.power_rating * self.charging[i],
                f"charge_{i}",
            )
            self.model.addConstr(
                self.power_discharge[i] <= self.power_rating * (1 - self.charging[i]),
                f"discharge_{i}",
            )

            # Solar to battery constraints
            self.model.addConstr(
                self.power_solar_to_battery[i] <= self.solar_generation[i],
                f"solar_to_battery_{i}",
            )
            self.model.addConstr(
                self.power_solar_to_battery[i] + self.battery_power_charge[i]
                <= self.power_rating * self.charging[i],
                f"solar_and_battery_charge_{i}",
            )

            # Grid connection constraints
            self.model.addConstr(
                self.solar_generation[i] + self.power_discharge[i]
                <= self.grid_connection_capacity,
                f"grid_connection_{i}",
            )

            # Battery SOC dynamics
            power_charge = (
                self.eta_charge * self.battery_power_charge[i]
                + self.power_solar_to_battery[i]
            )
            power_discharge = self.power_discharge[i] / self.eta_discharge
            self.model.addConstr(
                self.battery_soc[i + 1]
                == self.battery_soc[i] + power_charge - power_discharge,
                f"soc_dynamics_{i}",
            )

    def run_optimization(self):
        """Run Gurobi optimization."""
        self.logger.info("Running optimization...")
        self.model.optimize()

    def get_results(self) -> dict:
        """Get enhanced optimization results with additional insights."""

        # Extract results
        prices = np.array(self.prices)
        power = np.round(
            [
                self.battery_power_charge[i].x - self.power_discharge[i].x
                for i in range(self.hours_per_day)
            ],
            2,
        )
        e_soc = np.round([self.battery_soc[i].x for i in range(self.hours_per_day)], 2)

        # Calculate profit based directly on the optimization model
        profit = np.round(
            prices
            * np.array(
                [
                    self.power_discharge[i].x - self.battery_power_charge[i].x
                    for i in range(self.hours_per_day)
                ]
            ),
            2,
        )

        solar_power_to_grid = np.round(
            self.solar_generation
            - np.array(
                [self.power_solar_to_battery[i].x for i in range(self.hours_per_day)]
            ),
            2,
        )

        # Additional metrics
        charge_power = [
            self.battery_power_charge[i].x for i in range(self.hours_per_day)
        ]
        discharge_power = [self.power_discharge[i].x for i in range(self.hours_per_day)]

        # Instead of manually calculating Total Profit, use the best objective directly
        total_profit = (
            self.model.ObjVal
        )  # This ensures consistency between profit and objective

        # Time index
        time_index = pd.date_range(start="00:00", periods=self.hours_per_day, freq="H")

        # Create DataFrame with results
        result_df = pd.DataFrame(
            data=np.array(
                [
                    power,
                    e_soc,
                    solar_power_to_grid,
                    profit,
                    charge_power,
                    discharge_power,
                ]
            ).transpose(),
            columns=[
                "power",
                "e_soc",
                "solar_power_to_grid",
                "profit",
                "charge_power",
                "discharge_power",
            ],
            index=time_index,
        )

        # Extract Best Objective value from Gurobi (which represents the best possible profit)
        best_objective = self.model.ObjVal

        # Summary (optional)
        summary = {
            "Total Profit": total_profit,  # Use best objective as total profit to avoid mismatches
            "Best Objective": best_objective,
            "Total Solar to Grid": np.sum(solar_power_to_grid),
            "Total Energy Stored in Battery": np.sum(e_soc),
            "Average SOC": np.mean(e_soc),
            "Peak Discharge Power": np.max(discharge_power),
        }

        self.logger.info(f"Summary of results: {summary}")

        # Convert result_df to a list of records (suitable for JSON response)
        detailed_results = result_df.to_dict(orient="records")

        # Return both detailed results and the summary
        return {"summary": summary, "detailed_results": detailed_results}

    def plot_results_as_base64_old(self, result_df: pd.DataFrame):
        """Generate a plot and return it as a base64 string."""

        # Ensure the result_df is a pandas DataFrame
        if isinstance(result_df, list):
            result_df = pd.DataFrame(result_df)

        # Create the plot
        fig = make_subplots(
            rows=3, cols=1, subplot_titles=["Power Flow", "SOC", "Solar Power to Grid"]
        )

        fig.add_trace(go.Scatter(y=result_df["power"], name="Power Flow"), row=1, col=1)
        fig.add_trace(go.Scatter(y=result_df["e_soc"], name="SOC"), row=2, col=1)
        fig.add_trace(
            go.Scatter(y=result_df["solar_power_to_grid"], name="Solar to Grid"),
            row=3,
            col=1,
        )

        # Save the plot to a BytesIO buffer
        buffer = BytesIO()
        fig.write_image(buffer, format="png")
        buffer.seek(0)

        # Encode the image in base64 and return it as a string
        plot_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()

        return plot_base64

    def plot_results_as_base64(self, result_df):
        """Generates multiple plots and returns them as base64 encoded string."""

        # Ensure the result_df is a pandas DataFrame
        if isinstance(result_df, list):
            result_df = pd.DataFrame(result_df)

        fig = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=[
                "Battery SOC Over Time",
                "Profit Over Time",
                "Cumulative Energy Flows",
                "Charge vs. Discharge Patterns",
                "Grid Import/Export",
            ],
        )

        # 1. Battery SOC Over Time
        fig.add_trace(
            go.Scatter(y=result_df["e_soc"], name="Battery SOC"), row=1, col=1
        )

        # 2. Profit Over Time
        fig.add_trace(go.Scatter(y=result_df["profit"], name="Profit"), row=2, col=1)

        # 3. Cumulative Energy Flows (Solar Generation vs Battery Storage)
        cumulative_solar = np.cumsum(result_df["solar_power_to_grid"])
        cumulative_storage = np.cumsum(
            result_df["charge_power"] - result_df["discharge_power"]
        )
        fig.add_trace(
            go.Scatter(y=cumulative_solar, name="Cumulative Solar to Grid"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=cumulative_storage, name="Cumulative Battery Energy"),
            row=3,
            col=1,
        )

        # 4. Charge vs Discharge Patterns
        fig.add_trace(
            go.Bar(y=result_df["charge_power"], name="Battery Charge"), row=4, col=1
        )
        fig.add_trace(
            go.Bar(y=result_df["discharge_power"], name="Battery Discharge"),
            row=4,
            col=1,
        )

        # 5. Grid Import/Export
        fig.add_trace(
            go.Scatter(y=result_df["solar_power_to_grid"], name="Grid Export"),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=result_df["power"], name="Net Power Flow"), row=5, col=1
        )

        fig.update_layout(
            height=1000, title_text="GridEdgeOptimizer Comprehensive Analysis"
        )

        # Convert the plot to a base64 string to send as part of API response
        img_bytes = fig.to_image(format="png")
        buffer = BytesIO(img_bytes)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return plot_base64

    def plot_results(self, result_df):
        """Plot the optimization results."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1, subplot_titles=["Power Flow", "SOC", "Solar Power to Grid"]
        )

        fig.add_trace(go.Scatter(y=result_df["power"], name="Power Flow"), row=1, col=1)
        fig.add_trace(go.Scatter(y=result_df["e_soc"], name="SOC"), row=2, col=1)
        fig.add_trace(
            go.Scatter(y=result_df["solar_power_to_grid"], name="Solar to Grid"),
            row=3,
            col=1,
        )

        fig.show()

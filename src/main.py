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


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .grid_edge_optimizer import GridEdgeOptimizer


class BatteryConfig(BaseModel):
    soc_min: float
    soc_max: float
    capacity: float
    power_rating: float
    eta_charge: float
    eta_discharge: float


class SolarConfig(BaseModel):
    grid_connection_capacity: float


class OptimizationConfig(BaseModel):
    hours_per_day: int
    prices: List[float]
    solar_generation: List[float]


class GurobiConfig(BaseModel):
    threads: int


class GridEdgeRequest(BaseModel):
    battery: BatteryConfig
    solar: SolarConfig
    optimization: OptimizationConfig
    gurobi: GurobiConfig


app = FastAPI()


@app.post("/optimize/")
def optimize_solar_and_storage(data: GridEdgeRequest):
    optimizer = GridEdgeOptimizer(data.dict())
    optimizer.run_optimization()

    # Get optimization results and summary
    results = optimizer.get_results()

    # Generate plot as base64
    plot_base64 = optimizer.plot_results_as_base64(results["detailed_results"])

    # Return summary, detailed results, and the plot (encoded in base64)
    return {
        "summary": results["summary"],
        "detailed_results": results["detailed_results"],
        "plot_base64": plot_base64,  # Base64-encoded plot image
    }

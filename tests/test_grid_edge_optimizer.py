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


import unittest
import json
from src.grid_edge_optimizer import GridEdgeOptimizer


class TestGridEdgeOptimizer(unittest.TestCase):
    def setUp(self):
        # Load the test configuration from JSON
        with open("tests/test_config.json", "r") as f:
            config = json.load(f)
        self.optimizer = GridEdgeOptimizer(config)
        self.optimizer.run_optimization()

    def test_soc_limits(self):
        # Get the detailed results (not the summary)
        results = self.optimizer.get_results()["detailed_results"]

        # Ensure the SOC values are within valid limits
        e_soc_values = [item["e_soc"] for item in results]
        self.assertTrue(
            all(0 <= soc <= self.optimizer.battery_capacity for soc in e_soc_values)
        )

    def test_optimization_results(self):
        # Get the results and ensure they are not empty
        results = self.optimizer.get_results()["detailed_results"]
        self.assertTrue(len(results) > 0)

        # Check if the results contain "profit" key
        self.assertIn("profit", results[0])

    def test_summary(self):
        # Test the summary to ensure that all key metrics are present
        summary = self.optimizer.get_results()["summary"]
        self.assertIn("Total Profit", summary)
        self.assertIn("Best Objective", summary)
        self.assertIn("Total Solar to Grid", summary)
        self.assertIn("Total Energy Stored in Battery", summary)


if __name__ == "__main__":
    unittest.main()

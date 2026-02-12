import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeVar:
    def __init__(self, values):
        self.value = list(values)


class FakeModel:
    def __init__(self, time):
        self.time = np.array(time, dtype=float)


@pytest.fixture
def model_vars_factory():
    def _factory(nhrs=6):
        time = np.arange(nhrs)
        ones = np.ones(nhrs)
        linear = np.linspace(1.0, 2.0, nhrs)
        return {
            "model": FakeModel(time),
            "load": FakeVar(ones * 10.0),
            "hydrogen_demand_g": FakeVar(ones * 2.0),
            "potable_water_demand_l": FakeVar(ones * 3.0),
            "batt_power_inout": FakeVar(np.linspace(-1.0, 1.0, nhrs)),
            "batt_storagelevel": FakeVar(ones * 12.0),
            "h2_power_in": FakeVar(ones * 4.0),
            "h2_storagelevel": FakeVar(ones * 5.0),
            "potable_water_power_in": FakeVar(ones * 6.0),
            "potable_water_storagelevel": FakeVar(ones * 7.0),
            "generator_unitfactor": FakeVar(ones * 0.5),
            "generator_scale": FakeVar(ones * 8.0),
            "wind_unitfactor": FakeVar(ones * 0.6),
            "wind_cur": FakeVar(ones * 0.8),
            "wind_scale": FakeVar(ones * 9.0),
            "solar_unitfactor": FakeVar(ones * 0.7),
            "solar_cur": FakeVar(ones * 0.9),
            "solar_scale": FakeVar(ones * 10.0),
            "wave_unitfactor": FakeVar(ones * 0.4),
            "wave_cur": FakeVar(ones * 0.7),
            "wave_scale": FakeVar(ones * 11.0),
            "avg_generator_W": FakeVar(linear * 10.0),
            "avg_wind_W": FakeVar(linear * 11.0),
            "avg_sol_W": FakeVar(linear * 12.0),
            "avg_wave_W": FakeVar(linear * 13.0),
            "total_cost": FakeVar(linear * 1234.0),
            "cost_per_watt": FakeVar(linear * 1.23),
            "total_mass": FakeVar(linear * 45.6),
            "supported_mass": FakeVar(linear * 40.0),
            "floating_platform_mass": FakeVar(linear * 5.0),
        }

    return _factory

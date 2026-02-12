# Theory Guide

This section summarizes the model equations, constraints, and optimization formulation used in the GEKKO implementation.

## Notation
- Time index: `t` (hours)
- Electrical load: `L(t)` [W]
- Battery power: `P_batt(t)` [W] (positive discharging, negative charging)
- Generator power: `P_gen(t)` [W]
- Wind power: `P_wind(t)` [W]
- Solar power: `P_solar(t)` [W]
- Wave power: `P_wave(t)` [W]
- Hydrogen production power: `P_H2(t)` [W]
- Potable water production power: `P_H2O(t)` [W]
- Hydrogen demand: `D_H2(t)` [g/hr], time-varying profile scaled from CSV input
- Potable water demand: `D_H2O(t)` [L/hr], time-varying profile scaled from CSV input
- Battery energy: `E_batt(t)` [Wh]
- Hydrogen storage: `S_H2(t)` [g]
- Potable water storage: `S_H2O(t)` [L]
- Supported mass: `M_support` [kg], sum of component masses excluding the platform
- Floating platform mass: `M_platform` [kg]

## Differential equations

Battery storage:
```
dE_batt/dt = - eta_batt * P_batt
```

Hydrogen storage:
```
dS_H2/dt = P_H2 / e_H2 - D_H2
```

Potable water storage:
```
dS_H2O/dt = P_H2O / e_H2O - D_H2O
```

Where:
- `eta_batt` is the battery round-trip efficiency
- `e_H2` is energy per gram of hydrogen
- `e_H2O` is energy per liter of potable water

## Power balance constraint
At each time step, electric demand includes production loads:
```
P_batt + P_gen + P_wind + P_solar + P_wave = L + P_H2 + P_H2O
```

Wind, solar, and wave power are limited by available resource signals and curtailment variables:
```
P_wind = Wind_unitfactor * wind_cur * wind_scale
P_solar = Solar_unitfactor * solar_cur * solar_scale
P_wave = Wave_unitfactor * wave_cur * wave_scale
```

## Storage constraints
- Battery storage bounds:
  - `0 <= E_batt <= batt_scale`
- Hydrogen storage bounds:
  - `0 <= S_H2 <= h2_storage_scale`
- Potable water storage bounds:
  - `0 <= S_H2O <= potable_water_storage_scale`
- Final battery charge constraint:
  - `E_batt(t_final) = batt_scale * batt_final_charge`

Storage scale variables are bounded by configured limits:
- `h2_storage_scale <= h2_max_storage_g`
- `potable_water_storage_scale <= potable_water_max_storage_l`

## Floating platform sizing
Supported mass excludes the floating platform itself:
```
M_support = generator_mass + wind_mass + solar_mass + wave_mass + battery_mass
           + h2_storage_mass + potable_water_storage_mass
```
Platform mass scales with supported mass:
```
M_platform = alpha_platform * M_support
```
Where `alpha_platform` is `floating_platform_mass_per_supported_mass`.

## Objective functions

Two objectives are supported:

1) Total mass
```
min total_mass =
  supported_mass + platform_mass
```

2) Cost per watt
```
min cost_per_watt =
  (fixed_costs + variable_costs) / average_load

Fixed costs include generator, wind, solar, wave, battery, hydrogen storage, and potable water storage.
Platform cost is proportional to platform mass.
```

Average load and average production values are computed via integrals over the horizon using the full electric demand.

## Optimization formulation
This is a dynamic optimization problem with:
- Continuous state variables (storage levels)
- Manipulated variables for dispatch and curtailment
- Fixed and free variables for sizing (capacity scaling)
- Equality constraints for dynamics and power balance

GEKKO solves the resulting nonlinear program (NLP) using IPOPT by default when remote solving is enabled.

## Notes on GEKKO and IPOPT
- GEKKO is a Python modeling language for dynamic optimization; it builds NLPs from algebraic and differential equations.
  Documentation: https://gekko.readthedocs.io/
- IPOPT (Interior Point OPTimizer) solves large-scale nonlinear optimization problems using interior-point methods.
  Documentation: https://coin-or.github.io/Ipopt/

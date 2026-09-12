# Theory Guide

The model chooses continuous generation capacities, storage capacities, and hourly dispatch to meet a prescribed mission. Wind, solar, and wave enter as normalized electrical-availability profiles. Hydrogen production and desalination consume electricity and fill inventories from which service demand is withdrawn.

This is an architecture screen with simplified component properties. It does not resolve device hydrodynamics, battery degradation, generator transients, fuel inventory, water treatment quality, structural loads, or site suitability.

## Time and input boundary

The grid is `t = 0, 1, ..., nhrs-1` hours. GEKKO uses two collocation nodes per interval (`NODES=2`), giving a backward-Euler storage update. Row zero initializes storage; the subsequent rows describe the preceding one-hour intervals. Thus `nhrs` samples span `T = nhrs-1` solved hours. Only those intervals enter reported averages and feasibility checks.

Each selected CSV profile is divided by its own maximum; an all-zero profile stays zero. Non-fuel outage masks then remove availability without rescaling unaffected hours. Electrical demand is the normalized load shape times `peak_load`, plus constant `hotel_load`. The hydrogen and water shapes are multiplied by the historical `H2DailyDemand` and `H2ODailyDemand` settings, which are peak hourly rates in g/h and L/h. Daily totals must be computed by integrating those rates.

Outage start/stop settings select sample indices; fractional bounds round outward.
An outage starting at sample 8 for three samples disables rows 8–10, whose
dispatch describes elapsed hours 7–10. Plot shading follows these solved
intervals; row zero is only the initial state.

## Electrical and inventory balances

Positive battery bus power means discharge. Let `P_ch` and `P_dis` be nonnegative charge and discharge powers in W, and let `eta = sqrt(batt_round_trip_eff)`. Over each interval of length `dt = 1 h`:

```text
P_batt = P_dis - P_ch
E[k] - E[k-1] = dt * (eta * P_ch[k] - P_dis[k] / eta)
P_ch[k] * P_dis[k] = 0
```

The last equality prevents simultaneous charging and discharging. Both powers obey `batt_max_charge_rate`. For example, 100 Wh charged at a round-trip efficiency of 0.81 can subsequently deliver 81 Wh to the bus when the battery returns to its starting energy.

Hydrogen and potable-water inventories use production energy intensities `e_H2` in Wh/g and `e_water` in Wh/L:

```text
H[k] - H[k-1] = dt * (P_H2[k] / e_H2 - D_H2[k])
W[k] - W[k-1] = dt * (P_water[k] / e_water - D_water[k])
```

`D_H2` is in g/h and `D_water` is in L/h. Production obeys hard rate bounds; a zero bound disables production. Water's configured US-gallon/minute limit is multiplied by `60 * 3.785411784` to obtain L/h.

Hydrogen and water provide material service; this model has no fuel cell or other path returning their stored material to the electrical bus. Their plotted equivalent Wh are the electricity required to produce the inventory, not recoverable electrical energy. Where curves are multiplied for visibility, the legend states the multiplier.

The common electrical bus must satisfy:

```text
P_dis - P_ch + P_gen + P_wind + P_solar + P_wave
    = Load + P_H2 + P_water
```

Generation dispatch is nonnegative. The generator is bounded by installed capacity; each non-fuel source is bounded by its availability times installed capacity. Exported `wind_cur`, `solar_cur`, and `wave_cur` are the **fraction of available power used**, where 1 means no curtailment and 0 means no output. Fractions are reported as zero when availability is zero.

## Capacity and storage constraints

Installed sizes lie between configured minimum and maximum capacities. Equal limits fix a size. Unchecked GUI sizing controls set the maximum equal to the minimum before model construction; CLI reads the same controls.

```text
batt_min_storage_level <= E[k] <= batt_scale
0 <= H[k] <= h2_storage_scale
0 <= W[k] <= potable_water_storage_scale
E[0] = batt_initial_charge * batt_scale
E[end] = batt_final_charge * batt_scale
H[0] = h2_initial_charge * h2_storage_scale
W[0] = potable_water_initial_charge * potable_water_storage_scale
```

All initial fractions and the battery final fraction lie in [0, 1]. Initial battery energy must also satisfy its minimum storage level. Hydrogen and water have no terminal replenishment requirement. Their supplied initial inventory may be depleted. This boundary describes a finite mission and does not establish indefinitely repeatable operation.

## Mass and cost boundary

Component masses are proportional to installed capacities using kg/W, kg/Wh, kg/g, or kg/L coefficients. The supported equipment and platform are:

```text
M_support = M_generator + M_wind + M_solar + M_wave + M_battery + M_H2_storage
M_platform = floating_platform_mass_per_supported_mass * M_support
M_total = M_support + M_platform + M_water_storage
```

The potable-water tank is modeled as independently supported: its mass enters total mass but does not increase platform mass. Stored fluid mass, fuel mass, and separate production-equipment ratings are not resolved. These assumptions must be checked for a particular installation.

Total cost is installed-capacity cost plus platform cost and operating cost extrapolated over `lifespan` hours. The extrapolation uses each generation source's mean dispatch over the solved intervals; battery variable cost uses mean charging plus discharging throughput. Diesel variable cost includes fuel cost and a refill-service allocation divided by tank electrical energy (`tank_size * diesel_energy * gen_eff`). This is a simplified lifetime cost estimate without discounting, replacement schedules, or a fuel-stock constraint.

Default cost and mass coefficients are illustrative screening assumptions, not current vendor quotations or measured properties of a complete installation. Replace them with traceable project inputs before drawing an economic or structural design conclusion.

## Objective choice

| Objective | Quantity minimized | Interpretation |
| --- | --- | --- |
| `total_mass` | Total modeled mass, kg | Lightest feasible modeled configuration |
| `total_cost` | Total modeled lifetime cost, $ | Least-cost feasible modeled configuration |
| `cost_per_watt` | Total cost / mean electrical demand, $/W | Cost ratio; denominator includes optimized hydrogen and water production loads |
| `cost_per_mass` | Total cost / total modeled mass, $/kg | Cost-to-mass ratio |

Ratio denominators include a small `1e-6` floor to avoid division by zero. `cost_per_watt` is not a levelized energy cost in $/kWh. Its denominator can increase through extra production and stored inventory; `cost_per_mass` can reward extra inexpensive mass. Use `total_cost` or `total_mass` for a direct design comparison unless the ratio itself is the intended question, and inspect capacity bounds and final inventory before accepting a ratio result.

## Numerical verification and limits

The default local solve uses APOPT; selecting solver 2 uses BPOPT. Battery complementarity and ratio objectives make the problem nonlinear; `solver.linear` must remain false when battery dispatch is enabled. A successful solve is a feasible local candidate, not proof of a global optimum.

After solving, the application checks electrical balance, each inventory update, production/capacity/storage bounds, initial conditions, the battery terminal condition, and cost/mass accounting in physical units. The diagnostic record includes residuals and tolerances. These checks detect broken assembly or a rejected numerical result. Analytic tests separately establish expected energy delivery, inventory drawdown, and minimum capacity in simple cases; they do not provide experimental validation or timestep convergence for arbitrary inputs.

[GEKKO's model methods](https://gekko.readthedocs.io/en/latest/model_methods.html) describe time discretization, variables, equations, and local/remote solves. The equations and conventions above describe this application's specific formulation.

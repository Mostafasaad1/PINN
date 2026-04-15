# Traditional MBD Simulation

## Overview
This folder contains traditional numerical implementations without neural networks, serving as a baseline for comparison.

## Files
| File | Description |
|------|-------------|
| [`Basic_sim_no_pinn.py`](Basic_sim_no_pinn.py) | Standard ODE solver |
| [`Basic_sim_no_pinn_animated.py`](Basic_sim_no_pinn_animated.py) | Animated visualization |
| [`MBD_Systems_Analysis.md`](MBD_Systems_Analysis.md) | Mathematical derivation |

## MBD Framework
The Differential-Algebraic Equation (DAE) formulation:
```
[M    Φ_q^T] [q̈]   = [F_e]
[Φ_q    0  ] [λ ]     [γ_c]
```

Where:
- `M`: Mass matrix
- `Φ_q`: Constraint Jacobian
- `λ`: Lagrange multipliers (reaction forces)
- `F_e`: External forces
- `γ_c`: Acceleration constraint RHS

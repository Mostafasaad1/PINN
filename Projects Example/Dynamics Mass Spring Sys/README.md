# Dynamics Mass Spring Systems: MBD Analysis

## Overview
This folder compares traditional Multibody Dynamics (MBD) simulation with PINN-based approaches for mass-spring-damper systems.

## Subfolders

### [No_PINN](No_PINN/)
Traditional numerical simulation without neural networks:
- `Basic_sim_no_pinn.py` - Standard ODE solver implementation
- `Basic_sim_no_pinn_animated.py` - Animated visualization
- `MBD_Systems_Analysis.md` - Mathematical derivation of MBD formulation

### [PINN](PINN/)
Neural network-based DAE solvers:
- **EX1 Pure Mass System**: Constrained mass on a 2D plane
- **EX2 Mass Spring System**: Mass with spring attachment
- **EX3 Mass Spring System damper**: Full mass-spring-damper with comparison plots

## Key Concepts
- Differential-Algebraic Equations (DAE)
- Constraint Jacobian matrices
- Lagrange multipliers (reaction forces)
- PINN as DAE residual minimizer

## Mathematical Framework
The MBD formulation uses:
```
[M    Φ_q^T] [q̈]   = [F_e]
[Φ_q    0  ] [λ ]     [γ_c]
```

## Running
```bash
# Traditional simulation
python No_PINN/Basic_sim_no_pinn.py

# PINN approaches
python PINN/EX1\ Pure\ Mass\ System/pinn_mbd_pure_mass.py
python PINN/EX2\ Mass\ Spring\ System/pinn_mbd_system2.py
python PINN/EX3\ Mass\ Spring\ System\ damper/pinn_mbd_system3.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

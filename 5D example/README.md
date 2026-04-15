# 5D PINN Example: Parametric Surrogate Model

## Overview
This folder demonstrates a parametric PINN that creates a surrogate model for battery thermal management across different materials. The 5th dimension represents material thermal diffusivity.

## Problem Description
Designing battery pack cooling for EVs by testing materials ranging from plastics to copper:
- **Inputs**: 5D space `(x, y, z, t, α)` where α is thermal diffusivity
- **Output**: Temperature `T(x, y, z, t, α)`

One neural network replaces hundreds of ANSYS/COMSOL simulations.

## Files
| File | Description |
|------|-------------|
| [`pinn_5d_parametric_surrogate.py`](pinn_5d_parametric_surrogate.py) | Parametric PINN implementation |

## Key Concepts
- Parametric PDE solving with material properties as inputs
- Surrogate model generation
- Design space exploration without re-simulation
- Universal approximator for material physics

## Running the Example
```bash
python pinn_5d_parametric_surrogate.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

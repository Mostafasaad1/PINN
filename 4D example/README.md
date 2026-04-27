# 4D PINN Example: 3D Heat Equation

## Overview
This folder introduces 4D PINNs solving the 3D heat equation over time. This is the first example dealing with partial differential equations (PDEs) rather than ODEs.

## Problem Description
Simulating heat transfer in a copper heatsink for GPU thermal management:
- **Inputs**: 4D spacetime `(x, y, z, t)`
- **Output**: Temperature `T(x, y, z, t)`

The temperature must satisfy the 3D Heat Equation:
```
∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
```

## Files
| File | Description |
|------|-------------|
| [`pinn_4d_heat_equation.py`](pinn_4d_heat_equation.py) | 4D PINN for heat transfer |
| `pinn_4d_results.png` | Temperature distribution visualization |

## Key Concepts
- Mesh-free PDE solving (no FEA grid required)
- Continuous function learning
- Boundary and initial condition enforcement
- Partial derivatives via automatic differentiation

## Running the Example
```bash
python pinn_4d_heat_equation.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

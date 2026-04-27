# 3D PINN Example: Paper Airplane Trajectory

## Overview
This folder demonstrates PINNs in 3D space, modeling a paper airplane affected by gravity and crosswind. The network predicts position in all three spatial dimensions.

## Problem Description
A paper airplane is thrown at the park, moving in 3D space:
- **X (Forward)**: No acceleration (`d²x/dt² = 0`)
- **Y (Sideways)**: Wind acceleration (`d²y/dt² = 2.0 m/s²`)
- **Z (Vertical)**: Gravity (`d²z/dt² = −9.81 m/s²`)

## Files
| File | Description |
|------|-------------|
| [`pinn_3d_airplane.py`](pinn_3d_airplane.py) | 3D PINN implementation with wind effects |
| `pinn_3d_results.png` | 3D trajectory visualization |

## Key Concepts
- 3D output neural network: `NN(t) → [x(t), y(t), z(t)]`
- Multiple independent physics constraints
- 3D visualization with `mpl_toolkits.mplot3d`

## Running the Example
```bash
python pinn_3d_airplane.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

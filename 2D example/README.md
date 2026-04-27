# 2D PINN Example: Projectile Trajectory

## Overview
This folder extends the PINN concept to 2D space, solving a projectile motion problem. The neural network learns to predict both X and Y coordinates of a block tossed by a robot arm.

## Problem Description
A toy block is tossed across a room, moving in two dimensions simultaneously:
- **X (Horizontal)**: Constant velocity motion (`d²x/dt² = 0`)
- **Y (Vertical)**: Gravitational acceleration (`d²y/dt² = −g`)

## Files
| File | Description |
|------|-------------|
| [`pinn_2d_trajectory.py`](pinn_2d_trajectory.py) | 2D PINN implementation |
| `pinn_2d_results.png` | Trajectory visualization results |

## Key Concepts
- Multi-output neural network: `NN(t) → [x(t), y(t)]`
- Coupled physics constraints in 2D
- Learning from sparse 2D trajectory data (15 frames)

## Running the Example
```bash
python pinn_2d_trajectory.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

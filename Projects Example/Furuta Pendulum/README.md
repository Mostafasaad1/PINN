# Furuta Pendulum: Swing-Up Trajectory Optimization

## Overview
This project uses PINNs for non-linear trajectory optimization - discovering the optimal control sequence to swing a Furuta pendulum from hanging to upright.

## Problem Description
Swing the pendulum from resting (hanging down) to balanced (upright) in exactly 2.0 seconds.

## PINN Architecture
- **Input**: Time `t` (0 to 2.0s)
- **Outputs**: `[θ₁, θ₂, u]` (arm angle, pendulum angle, motor torque)
- **Physics**: Coupled Lagrangian dynamics

## Files
| File | Description |
|------|-------------|
| [`pinn_furuta_swingup.py`](pinn_furuta_swingup.py) | Swing-up trajectory optimization |
| `furuta_pendulum_results.png` | Trajectory and control results |

## Key Features
- Boundary conditions: Start hanging, end upright with zero velocity
- Effort loss: Penalizes excessive torque (`u²`)
- Open-loop control: Generates optimal u(t) sequence

## Running
```bash
python pinn_furuta_swingup.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

# 1D PINN Example: Ball Drop

## Overview
This folder contains a beginner-friendly Physics-Informed Neural Network (PINN) example that solves a 1D ball drop problem. The implementation demonstrates the core concepts of PINNs using simple gravitational physics.

## Problem Description
A ball is thrown upward from a height of 10m with an initial velocity of 5 m/s. The PINN learns to predict the ball's position at any time by combining:
- **Data Loss**: Matching sparse observations (only 10-100 training points)
- **Physics Loss**: Satisfying the ODE: `d²y/dt² = −g` (gravitational acceleration)

## Files
| File | Description |
|------|-------------|
| [`pinn_ball_drop.py`](pinn_ball_drop.py) | Main PINN implementation with detailed comments |
| [`pinn_ball_drop.ipynb`](pinn_ball_drop.ipynb) | Jupyter notebook version for interactive exploration |
| `pinn_results_*.png` | Training result visualizations |

## Key Concepts
- Neural network as a function approximator: `NN(t) → y(t)`
- Automatic differentiation for computing derivatives
- Physics-constrained learning with minimal data

## Running the Example
```bash
python pinn_ball_drop.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

# 6D PINN Example: Inverse Kinematics Solver

## Overview
This folder implements a PINN-based inverse kinematics (IK) solver for a 6-DOF articulated robot arm. The network learns to map target poses to joint angles while respecting forward kinematics constraints.

## Problem Description
Commanding an industrial 6-axis arm to reach a specific 6D pose (position + orientation):
- **Input**: Target 6D pose `(x, y, z, rx, ry, rz)`
- **Output**: 6 joint angles `(θ₁, θ₂, θ₃, θ₄, θ₅, θ₆)`

The physics loss enforces Denavit-Hartenberg (DH) forward kinematics.

## Files
| File | Description |
|------|-------------|
| [`pinn_6dof_ik_solver.py`](pinn_6dof_ik_solver.py) | 6-DOF IK solver implementation |

## Key Concepts
- Singularity-free IK solving
- Denavit-Hartenberg parameters
- Geometric constraint enforcement
- Alternative to Newton-Raphson iteration

## Running the Example
```bash
python pinn_6dof_ik_solver.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

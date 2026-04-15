# Beam-Ball Balancer: MIMO Control Planning

## Overview
This project demonstrates PINN-based trajectory synthesis for a ball and plate system - a classic underactuated MIMO (Multi-Input Multi-Output) control problem.

## Problem Description
Move a solid sphere from a corner of a 2×2 meter plate (x=1, y=1) to the center (x=0, y=0) in exactly 3.0 seconds, stopping perfectly.

## PINN Architecture
- **Input**: Time `t` (0 to 3.0s)
- **Outputs**: `[x, y, θ_x, θ_y]` (position + plate tilts)
- **Physics**: Rolling dynamics of a sphere on inclined plane

## Files
| File | Description |
|------|-------------|
| [`pinn_ball_plate.py`](pinn_ball_plate.py) | MIMO trajectory synthesis implementation |
| `Beam_Ball_results.png` | Generated trajectory visualization |

## Key Physics
```
x'' = (5/7) * g * sin(θ_y)
y'' = -(5/7) * g * sin(θ_x)
```

## Running
```bash
python pinn_ball_plate.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

# Magnetic Levitation: Non-Linear Control

## Overview
This project demonstrates PINN-based trajectory synthesis for magnetic levitation, handling the challenging 1/z² singularity in the dynamics.

## Problem Description
Move a steel ball from z=20mm up to z=5mm smoothly in 1.5 seconds while respecting the non-linear dynamics.

## PINN Architecture
- **Input**: Time `t` (0 to 1.5s)
- **Outputs**: `[z, i]` (air gap position, coil current)
- **Physics**: `m*z'' = m*g - C*(i/z)²`

## Files
| File | Description |
|------|-------------|
| [`pinn_maglev_control.py`](pinn_maglev_control.py) | Maglev trajectory synthesis |
| `maglev_results.png` | Position and current profiles |

## Key Innovation
**Architectural Prior**: Softplus activation bounds the output to prevent z=0 guesses, avoiding division-by-zero NaN explosions during training.

## Running
```bash
python pinn_maglev_control.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

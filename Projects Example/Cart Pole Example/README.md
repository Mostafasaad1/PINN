# Cart-Pole: System Identification

## Overview
This project demonstrates PINN-based system identification - discovering unknown physical parameters (masses) from noisy encoder data.

## Problem Description
Given 2 seconds of recorded cart-pole motion with noisy encoders and no spec sheet, discover:
- Cart mass (M)
- Pole mass (m)

## PINN Architecture
- **Input**: Time `t`
- **Outputs**: `[x, θ]` (cart position, pole angle)
- **Learnable Parameters**: `M_guess`, `m_guess` (updated during training!)

## Files
| File | Description |
|------|-------------|
| [`pinn_cartpole_sysid.py`](pinn_cartpole_sysid.py) | System identification implementation |
| `Cart_Pole_results.png` | Parameter discovery results |

## Key Innovation
The mass parameters are `nn.Parameter` objects, allowing the optimizer to update both network weights AND physical parameters simultaneously.

## Running
```bash
python pinn_cartpole_sysid.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib

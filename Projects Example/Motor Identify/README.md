# Motor Identify - System Identification for Electric Motors

Physics-Informed Neural Network for discovering motor parameters from operational data.

## Overview

System identification is the process of building mathematical models of dynamic systems from measured data. This project uses a PINN to perform inverse system identification, discovering physical motor parameters (resistance, inductance, flux linkage) purely from operational data without requiring specialized test procedures.

## The Physics

### PMSM Voltage Equations

**d-axis voltage equation:**
```
u_d = R_s * i_d + L_d * (di_d/dt) - ω * L_q * i_q
```

**q-axis voltage equation:**
```
u_q = R_s * i_q + L_q * (di_q/dt) + ω * (L_d * i_d + ψ)
```

Where:
- **R_s**: Stator resistance (Ω)
- **L_d, L_q**: d-axis and q-axis inductances (H)
- **ψ**: Permanent magnet flux linkage (Wb)
- **ω**: Electrical angular velocity (rad/s)

## Key Features

- **Inverse System Identification**: Discovers parameters from operational data
- **Learnable Parameters**: Rs, Ld, Lq, ψ as network parameters
- **Physics-Informed**: Enforces PMSM voltage equations
- **Data-Physics Fusion**: Combines sensor data with physics constraints
- **Real-World Data**: Uses Paderborn LEA motor dataset (or synthetic fallback)

## Files

- `pinn_motor_id.py` - Python script implementation
- `pinn_motor_id.ipynb` - Interactive Jupyter notebook

## Usage

### Python Script
```bash
python pinn_motor_id.py
```

### Jupyter Notebook
```bash
jupyter notebook pinn_motor_id.ipynb
```

## Requirements

- torch>=1.9.0
- numpy>=1.19.0
- matplotlib>=3.3.0
- scipy>=1.5.0
- pandas>=1.2.0
- kagglehub (optional, for real dataset)

### Install kagglehub for real data
```bash
pip install kagglehub
```

## Results

The PINN successfully:
- Discovers all four physical parameters in a single training run
- Enforces actual motor equations, not just curve fitting
- Works with normal operational data
- Provides robust estimation even with measurement noise

## Output

The script generates:
- `motor_identification_dashboard.png` - Visualization of parameter convergence and sensor tracking

## Engineering Applications

- **Motor Commissioning**: Automatically identify motor parameters during startup
- **Condition Monitoring**: Detect parameter changes indicating motor degradation
- **Adaptive Control**: Update controller parameters as motor characteristics change
- **Quality Control**: Verify motor parameters during manufacturing
- **Fault Detection**: Identify parameter deviations indicating faults

## Advantages

- **No Specialized Tests Required**: Works with normal operational data
- **Simultaneous Estimation**: Discovers all parameters in a single training run
- **Physics-Informed**: Enforces actual motor equations
- **Robust to Noise**: Physics constraints provide regularization
- **Differentiable**: Enables gradient-based optimization
- **Real-Time Capable**: Fast inference for online parameter estimation

## Data Sources

The notebook supports two modes:
1. **Real Data**: Downloads Paderborn LEA motor dataset from Kaggle (requires kagglehub)
2. **Synthetic Data**: Generates synthetic PMSM data for demonstration (no external dependencies)
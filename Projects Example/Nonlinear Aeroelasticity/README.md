# Nonlinear Aeroelasticity - Duffing Oscillator System Identification

Physics-Informed Neural Network for identifying structural parameters from vibration data.

## Overview

Nonlinear Aeroelasticity describes how flexible structures, like airplane wings, vibrate under extreme aerodynamic forces. During extreme bending, the metal begins to resist further deformation to prevent snapping—a phenomenon called "Structural Hardening." This project uses a PINN to discover both linear stiffness and nonlinear hardening coefficients from noisy sensor data.

## The Physics

### Duffing Oscillator

The Duffing equation adds a nonlinear cubic stiffness term to the classic mass-spring-damper:

```
m*x'' + c*x' + k*x + α*x³ = F_wind(t)
```

Where:
- **m**: Mass (kg)
- **c**: Damping coefficient (Ns/m)
- **k**: Linear stiffness (N/m)
- **α**: Nonlinear hardening coefficient (N/m³)
- **F_wind(t)**: Aerodynamic forcing function (N)

The cubic term (α*x³) creates the characteristic "pinched" peaks in the displacement waveform, representing structural hardening.

## Key Features

- **Inverse System Identification**: Discovers k and α from noisy data
- **Nonlinear Dynamics**: Captures structural hardening effects
- **Noise Filtering**: Simultaneously filters sensor noise
- **Learnable Parameters**: Both linear and nonlinear parameters as network parameters
- **Physics-Informed**: Enforces Duffing oscillator equation

## Files

- `pinn_nonlinear_aeroelasticity.py` - Python script implementation
- `pinn_nonlinear_aeroelasticity.ipynb` - Interactive Jupyter notebook

## Usage

### Python Script
```bash
python pinn_nonlinear_aeroelasticity.py
```

### Jupyter Notebook
```bash
jupyter notebook pinn_nonlinear_aeroelasticity.ipynb
```

## Requirements

- torch>=1.9.0
- numpy>=1.19.0
- matplotlib>=3.3.0
- scipy>=1.5.0

## Results

The PINN successfully:
- Discovers both linear stiffness (k) and nonlinear hardening (α)
- Captures the characteristic "pinched" waveform
- Filters 5% Gaussian noise from sensor data
- Converges from poor initial guesses to accurate parameters

## Output

The script generates:
- `nonlinear_aeroelasticity_identification.png` - Visualization of displacement tracking and parameter convergence

## Engineering Applications

- **Aerospace Design**: Identify wing structural properties from flight test data
- **Condition Monitoring**: Detect parameter changes indicating structural degradation
- **Flutter Analysis**: Understand nonlinear aeroelastic behavior for safety certification
- **Wind Tunnel Testing**: Reduce test time by identifying parameters from limited data
- **Structural Health Monitoring**: Track changes in stiffness and hardening over time

## Advantages

- **No Specialized Tests Required**: Works with normal operational data
- **Simultaneous Estimation**: Discovers all parameters in a single training run
- **Physics-Informed**: Enforces actual structural dynamics
- **Noise Robust**: Physics constraints provide regularization
- **Differentiable**: Enables gradient-based optimization
- **Real-Time Capable**: Fast inference for online parameter estimation

## System Complexity

The Duffing oscillator is a 1-DOF nonlinear system with:
- 1 state variable (displacement x)
- 2 physical parameters (k, α)
- 1 nonlinear differential equation
- Rich dynamics including multiple equilibrium points, period-doubling bifurcations, and chaotic behavior

This makes it an ideal testbed for physics-informed machine learning methods.
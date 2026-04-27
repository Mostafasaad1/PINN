# Ballbot Balancer - 3D Inverted Pendulum Control

Physics-Informed Neural Network for controlling a 3D ballbot (inverted pendulum on a spherical wheel).

## Overview

A Ballbot is an inherently unstable 3D inverted pendulum balancing on a single spherical wheel. This project uses a PINN to discover optimal control torques that drive the robot from an unstable initial tilt back to equilibrium, respecting the full nonlinear physics.

## The Physics

The ballbot is modeled as a nonlinear inverted pendulum:

```
I * α - m*g*l*sin(θ) + Torque = 0
```

Where:
- **I**: Moment of inertia
- **m**: Mass of the ballbot body
- **l**: Distance from ball center to center of mass
- **θ**: Tilt angle (pitch and roll)
- **α**: Angular acceleration
- **g**: Gravity

## Key Features

- **Nonlinear Dynamics**: Uses full sin(θ) instead of linearized approximation
- **MIMO Control**: Simultaneous control of pitch and roll axes
- **NMPC Approach**: Acts as a Nonlinear Model Predictive Controller
- **Energy-Efficient**: Includes control effort penalty

## Files

- `pinn_ballbot_balancer.py` - Python script implementation
- `pinn_ballbot_balancer.ipynb` - Interactive Jupyter notebook

## Usage

### Python Script
```bash
python pinn_ballbot_balancer.py
```

### Jupyter Notebook
```bash
jupyter notebook pinn_ballbot_balancer.ipynb
```

## Requirements

- torch>=1.9.0
- numpy>=1.19.0
- matplotlib>=3.3.0

## Results

The PINN successfully:
- Learns the full nonlinear dynamics without linearization
- Generates smooth control torques from unstable initial conditions
- Converges to equilibrium with zero velocity
- Respects the coupled 3D motion physics

## Output

The script generates:
- `ballbot_pinn_control.png` - Visualization of learned trajectories and control torques

## Engineering Applications

- **Robotics**: Balancing robots and inverted pendulum systems
- **Control Systems**: Nonlinear model predictive control
- **Aerospace**: Attitude control for spacecraft
- **Biomechanics**: Human balance modeling
# Physics-Informed Neural Networks (PINN) Examples

A comprehensive collection of Physics-Informed Neural Network implementations, progressing from simple 1D problems to complex 6D robotics applications.

## Overview

This repository demonstrates how PINNs combine neural networks with physical laws to solve differential equations with minimal data. Each example is self-contained with detailed comments explaining the physics and implementation.

## Directory Structure

```
PINN/
└── Projects Example/           # Real-world applications
    ├── Ballbot Balancer/       # 3D inverted pendulum control
    ├── Beam-Ball Balancer/     # MIMO control planning
    ├── Cart Pole Example/      # System identification
    ├── Delta Robot Kinematics Solver/  # 6D inverse kinematics
    ├── Dynamics Mass Spring Sys/
    │   ├── No_PINN/           # Traditional MBD simulation
    │   └── PINN/              # Neural DAE solvers
    ├── Furuta Pendulum/        # Swing-up optimization
    ├── Magnetic Levitation Example/  # Non-linear control
    ├── Motor Identify/         # Motor parameter identification
    ├── Motor Observer (The Soft Sensor)/  # Digital twin for motors
    ├── Nonlinear Aeroelasticity/  # Duffing oscillator identification
    └── Reaction Wheel Lyapunov/  # Lyapunov-based control
```

## Requirements

### Core Dependencies
```
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.5.0
pandas>=1.2.0
```

### Installation
```bash
pip install torch numpy matplotlib scipy pandas
```

### For Jupyter Notebooks
```bash
pip install jupyter notebook
```

## Key Concepts

### What is a PINN?
A Physics-Informed Neural Network combines:
1. **Data Loss**: Matches neural network outputs to observed data
2. **Physics Loss**: Enforces physical laws (ODEs/PDEs) via automatic differentiation

### Why PINNs?
- **Mesh-free**: No need for finite element grids
- **Data-efficient**: Learn from sparse observations
- **Continuous**: Query any point in the domain
- **Physics-guaranteed**: Solutions respect physical laws

## Project Examples

### Ballbot Balancer
3D inverted pendulum control using nonlinear dynamics:
```bash
cd "Projects Example/Ballbot Balancer"
jupyter notebook pinn_ballbot_balancer.ipynb
# or
python pinn_ballbot_balancer.py
```

### Motor Observer (Soft Sensor)
Digital twin for estimating internal motor temperature:
```bash
cd "Projects Example/Motor Observer (The Soft Sensor)"
jupyter notebook pinn_motor_observer.ipynb
# or
python pinn_motor_observer.py
```

### Motor Identify
System identification for PMSM motor parameters:
```bash
cd "Projects Example/Motor Identify"
jupyter notebook pinn_motor_id.ipynb
# or
python pinn_motor_id.py
```

### Nonlinear Aeroelasticity
Duffing oscillator parameter identification:
```bash
cd "Projects Example/Nonlinear Aeroelasticity"
jupyter notebook pinn_nonlinear_aeroelasticity.ipynb
# or
python pinn_nonlinear_aeroelasticity.py
```

### System Identification
Discover unknown physical parameters from noisy data:
```bash
cd "Projects Example/Cart Pole Example"
python pinn_cartpole_sysid.py
```

### Trajectory Optimization
Generate optimal control sequences:
```bash
cd "Projects Example/Furuta Pendulum"
python pinn_furuta_swingup.py
```

### Multibody Dynamics
Compare traditional vs PINN approaches:
```bash
cd "Projects Example/Dynamics Mass Spring Sys"
```

## Learning Path

1. **Beginner**: Start with `1D example/` - detailed comments explain PINN fundamentals
2. **Intermediate**: Progress through 2D-4D examples to see PDE solving
3. **Advanced**: Explore 5D-6D for parametric and robotics applications
4. **Applied**: Study `Projects Example/` for real-world engineering problems

## Notebook Format

Several projects now include Jupyter notebook versions (`.ipynb`) for interactive exploration:
- Ballbot Balancer
- Motor Observer (The Soft Sensor)
- Motor Identify
- Nonlinear Aeroelasticity

These notebooks provide:
- Step-by-step execution
- Interactive visualizations
- Detailed documentation
- Parameter analysis

## License

This project is provided for educational purposes.

## Contributing

Contributions welcome! Areas for expansion:
- Additional physics problems
- Alternative neural architectures
- Comparison with traditional solvers
- Visualization improvements
- More notebook conversions
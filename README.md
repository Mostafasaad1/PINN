# Physics-Informed Neural Networks (PINN) Examples

A comprehensive collection of Physics-Informed Neural Network implementations, progressing from simple 1D problems to complex 6D robotics applications.

## Overview

This repository demonstrates how PINNs combine neural networks with physical laws to solve differential equations with minimal data. Each example is self-contained with detailed comments explaining the physics and implementation.

## Directory Structure

```
PINN/
└── Projects Example/           # Real-world applications
    ├── Beam-Ball Balancer/     # MIMO control planning
    ├── Cart Pole Example/      # System identification
    ├── Dynamics Mass Spring Sys/
    │   ├── No_PINN/           # Traditional MBD simulation
    │   └── PINN/              # Neural DAE solvers
    ├── Furuta Pendulum/        # Swing-up optimization
    └── Magnetic Levitation Example/  # Non-linear control
```

## Requirements

### Core Dependencies
```
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
```

### Installation
```bash
pip install torch numpy matplotlib
```

## ~~Examples Progression~~ **REMOVED**

| Dimension | Example | Physics | Application |
|-----------|---------|---------|-------------|
| 1D | Ball Drop | `d²y/dt² = -g` | Gravitational motion |
| 2D | Trajectory | `d²x/dt² = 0, d²y/dt² = -g` | Projectile motion |
| 3D | Airplane | Wind + gravity | 3D flight path |
| 4D | Heat Equation | `∂T/∂t = α∇²T` | Thermal management |
| 5D | Parametric | Material properties | Surrogate modeling |
| 6D | IK Solver | DH kinematics | Robot control |

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

## License

This project is provided for educational purposes.

## Contributing

Contributions welcome! Areas for expansion:
- Additional physics problems
- Alternative neural architectures
- Comparison with traditional solvers
- Visualization improvements

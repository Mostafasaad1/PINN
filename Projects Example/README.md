# Projects Example: Applied PINN Applications

## Overview
This folder contains real-world engineering applications of Physics-Informed Neural Networks, demonstrating how PINNs can solve practical control systems, dynamics, and robotics problems.

## Subfolders

### [Beam-Ball Balancer](Beam-Ball%20Balancer/)
MIMO control planning for a ball and plate system - moving a sphere to center using plate tilts.

### [Cart Pole Example](Cart%20Pole%20Example/)
System identification for cart-pole dynamics - discovering unknown mass parameters from noisy data.

### [Dynamics Mass Spring Sys](Dynamics%20Mass%20Spring%20Sys/)
Multibody dynamics (MBD) analysis comparing traditional simulation with PINN approaches for mass-spring-damper systems.

### [Furuta Pendulum](Furuta%20Pendulum/)
Non-linear trajectory optimization for swing-up control of a Furuta pendulum.

### [Magnetic Levitation Example](Magnetic%20Levitation%20Example/)
Non-linear control synthesis for magnetic levitation with 1/z² singularity handling.

## Key Applications
- **System Identification**: Discovering physical parameters from data
- **Trajectory Optimization**: Generating optimal control sequences
- **Inverse Kinematics**: Singularity-free robot control
- **Multibody Dynamics**: DAE-constrained motion planning

## Requirements
- PyTorch
- NumPy
- Matplotlib

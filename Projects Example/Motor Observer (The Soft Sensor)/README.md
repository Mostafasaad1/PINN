# Motor Observer (The Soft Sensor) - Digital Twin for Electrical Machines

Physics-Informed Neural Network for estimating internal motor temperature without physical sensors.

## Overview

Modern high-performance electrical machines are pushed to their thermal limits. Measuring internal rotor winding temperature directly is difficult, expensive, and prone to sensor failure. This project implements a "Soft Sensor" (Digital Twin) using a PINN to estimate hidden internal temperature from easily measurable external signals.

## The Physics

### Coupled Electro-Thermal Dynamics

**Electrical Dynamics:**
```
V(t) = L*(di/dt) + R(T)*i(t) + K*w(t)
```

Where resistance is temperature-dependent:
```
R(T) = R0 * (1 + α * (T(t) - T_amb))
```

**Thermal Dynamics:**
```
C_th*(dT/dt) = P_loss - (T(t) - T_amb)/R_th
```

Where Ohmic power loss:
```
P_loss = R(T) * i(t)²
```

## Key Features

- **Soft Sensor**: Estimates hidden temperature without physical sensors
- **Electro-Thermal Coupling**: Captures temperature-dependent resistance
- **Data-Physics Fusion**: Combines sparse sensor data with dense physics constraints
- **Real-Time Estimation**: Fast inference suitable for online monitoring

## Files

- `pinn_motor_observer.py` - Python script implementation
- `pinn_motor_observer.ipynb` - Interactive Jupyter notebook

## Usage

### Python Script
```bash
python pinn_motor_observer.py
```

### Jupyter Notebook
```bash
jupyter notebook pinn_motor_observer.ipynb
```

## Requirements

- torch>=1.9.0
- numpy>=1.19.0
- matplotlib>=3.3.0

## Results

The PINN successfully:
- Estimates internal rotor temperature without direct measurements
- Captures electro-thermal coupling effects
- Filters noise from sensor measurements
- Provides real-time temperature estimates

## Output

The script generates:
- `motor_observer_digital_twin.png` - Visualization of current tracking and temperature estimation

## Engineering Applications

- **Thermal Protection**: Prevent motor burnout by monitoring internal temperature
- **Condition Monitoring**: Detect anomalies and predict failures
- **Optimal Control**: Use temperature estimates for torque optimization
- **Predictive Maintenance**: Schedule maintenance based on thermal stress
- **Design Optimization**: Understand thermal behavior for better motor design

## Advantages

- **No Physical Sensor Required**: Eliminates expensive and unreliable temperature sensors
- **Physics-Informed**: Enforces actual electro-thermal laws
- **Robust to Noise**: Physics constraints provide regularization
- **Differentiable**: Enables gradient-based optimization
- **Real-Time Capable**: Fast inference for online monitoring
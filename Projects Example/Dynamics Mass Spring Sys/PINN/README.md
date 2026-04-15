# PINN-Based MBD Solvers

## Overview
This folder contains PINN implementations that replace traditional DAE matrix solvers for constrained multibody dynamics.

## Examples

### EX1: Pure Mass System
Simplest case - a mass constrained to move horizontally on a 2D plane.
- Input: time `t`
- Output: `[x(t), y(t), λ(t)]` (position + constraint force)

### EX2: Mass-Spring System
Mass with spring attachment, demonstrating energy storage.

### EX3: Mass-Spring-Damper System
Complete system with damping, including comparison plots against numerical solutions.

## Key Innovation
PINNs minimize DAE residuals directly, avoiding matrix inversion and handling constraints through the loss function.

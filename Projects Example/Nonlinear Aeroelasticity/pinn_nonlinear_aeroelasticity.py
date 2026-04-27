import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.integrate import solve_ivp

"""
===========================================================================
PROJECT: Nonlinear Aeroelasticity (The Duffing Oscillator)
         Physics-Informed System Identification
===========================================================================

THE ENGINEERING CONCEPT:
Nonlinear Aeroelasticity describes how flexible structures, like airplane wings, 
vibrate under extreme aerodynamic forces (such as severe turbulence). 

Under normal conditions, a wing flexes smoothly and predictably, much like a 
standard linear spring (kx). However, during extreme bending, the metal begins 
to resist further deformation to prevent the wing from snapping. This physical 
phenomenon is called "Structural Hardening".

Mathematically, it is modeled by the Duffing Oscillator, which adds a nonlinear 
cubic stiffness term (\alpha x^3) to the classic mass-spring-damper equation:
    m*x'' + c*x' + k*x + \alpha*x^3 = F_wind(t)

THE PINN OBJECTIVE (SYSTEM IDENTIFICATION):
We are provided with noisy sensor data from a wing undergoing forced vibration 
in a wind tunnel. The wave isn't a smooth sine wave—the peaks are "pinched" 
because of the \alpha x^3 term snapping the wing back. 

We do NOT know the wing's Linear Stiffness (k) or its Structural Hardening (\alpha).
The PINN will simultaneously filter the noisy data and dynamically tune its own 
internal physics parameters (via PyTorch's Autograd) until the Duffing equation 
perfectly matches the measured reality.

INPUTS:
    - t : Time [seconds] (shape: N x 1)

OUTPUTS:
    - x : Predicted displacement [meters] (shape: N x 1)

LEARNABLE PHYSICS PARAMETERS:
    - k     : Linear Stiffness (Target: 25.0 N/m)
    - alpha : Nonlinear Hardening (Target: 5.0)

===========================================================================
"""

# =========================================================================
# 1. SYNTHETIC SENSOR DATA GENERATION (THE "WIND TUNNEL")
# =========================================================================
# We generate the "ground truth" data using a standard numerical solver, 
# then add noise to simulate real-world sensors. The PINN does not know 
# the true 'k' and 'alpha' used here.

# Known fixed parameters
m = 1.0       # Mass (kg)
c = 0.5       # Damping coefficient (Ns/m)
F0 = 10.0     # Wind force amplitude (N)
omega = 2.0   # Wind forcing frequency (rad/s)

# TRUE parameters (Hidden from the PINN)
TRUE_K = 25.0
TRUE_ALPHA = 5.0

def duffing_system(t, state):
    """Numerical ODE for the Duffing Oscillator to generate training data."""
    x, v = state
    # m*x'' + c*x' + k*x + alpha*x^3 = F0*cos(omega*t)
    dxdt = v
    dvdt = (F0 * np.cos(omega * t) - c * v - TRUE_K * x - TRUE_ALPHA * (x**3)) / m
    return [dxdt, dvdt]

# Generate 2 seconds of sensor data
t_sensor_np = np.linspace(0, 2, 400)
sol = solve_ivp(duffing_system, [0, 2], [0.0, 0.0], t_eval=t_sensor_np)

# Add 5% Gaussian noise to simulate imperfect physical sensors
noise = np.random.normal(0, 0.05 * np.std(sol.y[0]), size=sol.y[0].shape)
x_sensor_noisy = sol.y[0] + noise

# Convert to PyTorch tensors
t_train = torch.tensor(t_sensor_np, dtype=torch.float32).view(-1, 1)
x_train = torch.tensor(x_sensor_noisy, dtype=torch.float32).view(-1, 1)

# Requires grad for physics residual calculation
t_train.requires_grad = True


# =========================================================================
# 2. PINN ARCHITECTURE & PARAMETER ENCAPSULATION
# =========================================================================

class DuffingPINN(nn.Module):
    def __init__(self):
        super(DuffingPINN, self).__init__()
        
        # The Neural Network: Approximates the displacement x(t)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # -----------------------------------------------------------------
        # THE GOTCHA: Learnable Physics Parameters
        # Instead of hardcoding k and alpha, we register them as PyTorch 
        # Parameters. They will receive gradients during backpropagation.
        # We start with terrible guesses (k=1.0, alpha=0.0 - assuming linear).
        # -----------------------------------------------------------------
        self.k = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
    def forward(self, t):
        return self.net(t)


# =========================================================================
# 3. TRAINING LOOP & PHYSICS RESIDUAL MINIMIZATION
# =========================================================================

pinn = DuffingPINN()
optimizer = optim.Adam(list(pinn.net.parameters()) + [pinn.k, pinn.alpha], lr=0.005)

epochs = 3000

print("="*60)
print("STARTING SYSTEM IDENTIFICATION: DUFFING OSCILLATOR")
print("="*60)
print(f"Initial Guesses -> k: {pinn.k.item():.2f}, alpha: {pinn.alpha.item():.2f}")
print("-" * 60)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. Network Prediction
    x_pred = pinn(t_train)
    
    # 2. Data Loss (Supervised learning on the noisy sensor data)
    loss_data = torch.mean((x_pred - x_train)**2)
    
    # 3. Autograd: Calculate velocity (x') and acceleration (x'')
    x_dot = torch.autograd.grad(
        x_pred, t_train, 
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True
    )[0]
    
    x_ddot = torch.autograd.grad(
        x_dot, t_train, 
        grad_outputs=torch.ones_like(x_dot),
        create_graph=True
    )[0]
    
    # 4. Physics Loss (The Duffing Oscillator Residual)
    # Reconstructing the aerodynamic forcing function: F_wind(t) = F0 * cos(omega*t)
    F_wind = F0 * torch.cos(omega * t_train)
    
    # Residual = m*x'' + c*x' + k*x + \alpha*x^3 - F_wind
    physics_residual = (m * x_ddot) + (c * x_dot) + (pinn.k * x_pred) + (pinn.alpha * (x_pred**3)) - F_wind
    loss_physics = torch.mean(physics_residual**2)
    
    # Total Loss
    loss_total = loss_data + (0.1 * loss_physics) # Weight physics slightly lower to let data fit first
    
    # Backpropagation
    loss_total.backward()
    optimizer.step()
    
    # Logging
    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Total Loss: {loss_total.item():.6f} "
              f"| Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_physics.item():.6f}")
        print(f"         > Identified Parameters -> k: {pinn.k.item():.4f} (True: {TRUE_K}), "
              f"alpha: {pinn.alpha.item():.4f} (True: {TRUE_ALPHA})")

print("="*60)
print("SYSTEM IDENTIFICATION COMPLETE.")
print(f"Final Identified Linear Stiffness (k)     : {pinn.k.item():.4f} N/m")
print(f"Final Identified Structural Hardening (\alpha): {pinn.alpha.item():.4f}")
print("="*60)
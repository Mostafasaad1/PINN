"""
===============================================================================
Project: Motor Observer (The Soft Sensor) - Digital Twin for Electrical Machines
===============================================================================

The Engineering Concept:
Modern high-performance electrical machines (like PMSMs or heavy-duty DC motors) 
are pushed to their thermal limits to maximize torque density. However, measuring 
the internal rotor winding temperature directly is notoriously difficult, expensive, 
and prone to mechanical sensor failure. If the winding overheats, the insulation 
melts, causing catastrophic motor burnout.

This project implements a "Soft Sensor" (a Digital Twin) using a Physics-Informed 
Neural Network (PINN). Instead of relying on a physical temperature sensor, the 
PINN acts as a state observer. It reads the easily measurable external signals 
(Terminal Voltage and Phase Current) and uses the coupled Electro-Thermal 
Differential Equations to instantly "observe" and estimate the hidden internal 
rotor temperature in real-time.

The Physics (Coupled Electro-Thermal Dynamics):
1. Electrical Dynamics:
   V(t) = L*(di/dt) + R(T)*i(t) + K*w(t)
   Where R(T) is temperature-dependent: R(T) = R0 * (1 + alpha * (T(t) - T_amb))

2. Thermal Dynamics:
   C_th*(dT/dt) = P_loss - (T(t) - T_amb)/R_th
   Where Ohmic power loss P_loss = R(T) * i(t)^2

Outputs (Neural Network Predictions):
    i_pred : Estimated Current (Amperes)
    T_pred : Estimated Internal Temperature (Celsius) - The Hidden State!
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Physical Motor Parameters (Constants)
# =============================================================================
L      = 0.05    # Inductance (Henries)
R0     = 1.2     # Base phase resistance at ambient temperature (Ohms)
alpha  = 0.00393 # Temperature coefficient of copper (1/Celsius)
K_e    = 0.1     # Back-EMF constant (V/(rad/s))
C_th   = 50.0    # Thermal capacitance of the rotor (J/Celsius) - Heat capacity
R_th   = 2.5     # Thermal resistance to environment (Celsius/Watt) - Cooling rate
T_amb  = 25.0    # Ambient environment temperature (Celsius)

# =============================================================================
# 2. The Physics-Informed Neural Network (Digital Twin)
# =============================================================================
class MotorDigitalTwinPINN(nn.Module):
    def __init__(self):
        super(MotorDigitalTwinPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) 
        )
        
    def forward(self, t):
        out = self.net(t)
        i_pred = out[:, 0:1] * 20.0        # Scale output for current
        T_pred = T_amb + out[:, 1:2] * 100.0 # Scale output for temperature
        return i_pred, T_pred

# =============================================================================
# 3. Physics & Data Loss Formulation
# =============================================================================
def calculate_loss(pinn, t_collocation, t_data, V_data, w_data, i_data_measured):
    # A. Data Loss (Sensor Alignment - Current Only)
    i_pred_data, _ = pinn(t_data)
    loss_data = torch.mean((i_pred_data - i_data_measured)**2)
    
    # B. Physics Loss (Electro-Thermal ODEs)
    t_collocation.requires_grad_(True)
    i_pred, T_pred = pinn(t_collocation)
    
    di_dt = torch.autograd.grad(i_pred, t_collocation, grad_outputs=torch.ones_like(i_pred), create_graph=True)[0]
    dT_dt = torch.autograd.grad(T_pred, t_collocation, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    
    V_input = 24.0  # Constant nominal voltage
    w_input = 100.0 # Constant speed
    
    R_t = R0 * (1.0 + alpha * (T_pred - T_amb))
    
    electrical_residual = (L * di_dt) + (R_t * i_pred) + (K_e * w_input) - V_input
    
    heat_generated = (i_pred**2) * R_t
    heat_dissipated = (T_pred - T_amb) / R_th
    thermal_residual = (C_th * dT_dt) - heat_generated + heat_dissipated
    
    loss_physics = torch.mean(electrical_residual**2) + torch.mean(thermal_residual**2)
    total_loss = (10.0 * loss_data) + (1.0 * loss_physics)
    
    return total_loss, loss_data, loss_physics

# =============================================================================
# 4. Training Initialization & Execution
# =============================================================================
def train_motor_digital_twin():
    t_max = 600.0 # 10-minute heavy load cycle
    
    # Synthetic Sensor Data
    t_data_np = np.linspace(0, t_max, 200).reshape(-1, 1)
    i_steady_state = (24.0 - (K_e * 100.0)) / (R0 * 1.5) 
    i_data_measured_np = np.full_like(t_data_np, i_steady_state) 
    
    t_data = torch.tensor(t_data_np, dtype=torch.float32)
    i_data_measured = torch.tensor(i_data_measured_np, dtype=torch.float32)
    
    t_collocation_np = np.linspace(0, t_max, 1000).reshape(-1, 1)
    t_collocation = torch.tensor(t_collocation_np, dtype=torch.float32)

    V_data = torch.full_like(t_data, 24.0)
    w_data = torch.full_like(t_data, 100.0)

    pinn_digital_twin = MotorDigitalTwinPINN()
    optimizer = optim.Adam(pinn_digital_twin.parameters(), lr=1e-3)
    
    epochs = 5000
    
    # Tracking for Analysis
    history = {'total': [], 'data': [], 'phys': []}
    
    print("==========================================================")
    print("Initiating Motor Observer (Digital Twin) Training Phase...")
    print("==========================================================")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_data, loss_phys = calculate_loss(pinn_digital_twin, t_collocation, t_data, V_data, w_data, i_data_measured)
        loss.backward()
        optimizer.step()
        
        history['total'].append(loss.item())
        history['data'].append(loss_data.item())
        history['phys'].append(loss_phys.item())
        
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Total Loss: {loss.item():.6f} | Sensor Data: {loss_data.item():.6f} | Physics: {loss_phys.item():.6f}")

    return pinn_digital_twin, history, t_collocation_np, t_data_np, i_data_measured_np

# =============================================================================
# 5. Analysis & Results (Visualization)
# =============================================================================
def plot_results(pinn, history, t_col, t_data, i_measured):
    print("\nGenerating Analysis Dashboard...")
    
    # Generate continuous predictions for plotting
    t_tensor = torch.tensor(t_col, dtype=torch.float32)
    with torch.no_grad():
        i_pred, T_pred = pinn(t_tensor)
        
    i_pred_np = i_pred.numpy()
    T_pred_np = T_pred.numpy()
    
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title("PINN Motor Observer - Analysis")
    plt.suptitle("Soft Sensor Digital Twin Analysis", fontsize=16, fontweight='bold')
    
    # 1. Convergence History
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(history['total'], label='Total Loss', color='black', linewidth=2)
    ax1.plot(history['data'], label='Sensor Data Loss', color='blue', alpha=0.7)
    ax1.plot(history['phys'], label='Physics ODE Loss', color='red', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_title("Training Convergence (Log Scale)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # 2. Current Tracking (Ammeter Observation)
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t_col, i_pred_np, label='PINN Estimated Current', color='blue', linewidth=2)
    ax2.scatter(t_data[::5], i_measured[::5], color='black', label='Measured Sensor Data', zorder=5, s=15)
    ax2.set_title("External State Tracking (Phase Current)")
    ax2.set_ylabel("Current (Amperes)")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    # 3. Temperature Observation (The Hidden State)
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t_col, T_pred_np, label='PINN Inferred Rotor Temp', color='red', linewidth=2)
    ax3.axhline(y=T_amb, color='green', linestyle='--', label='Ambient Temp (25°C)')
    ax3.set_title("Hidden State Observation (Rotor Temperature)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(True, ls="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    trained_pinn, loss_history, t_collocation, t_data, i_measured = train_motor_digital_twin()
    plot_results(trained_pinn, loss_history, t_collocation, t_data, i_measured)
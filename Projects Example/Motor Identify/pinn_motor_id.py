import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

# ==========================================
# 1. KAGGLEHUB DATA IMPORT & PREPROCESSING
# ==========================================
print("📥 Downloading dataset using kagglehub...")
dataset_path = kagglehub.dataset_download("hankelea/system-identification-of-an-electric-motor")
print(f"✅ Download complete. Files located at: {dataset_path}")

# Dynamically find the CSV file in the downloaded directory
csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the downloaded dataset.")

# The LEA dataset usually uses 'measures_v2.csv' or 'pmsm_temperature_data.csv'
CSV_FILE = csv_files[0] 
print(f"📊 Loading sensor data from: {os.path.basename(CSV_FILE)}...")

df = pd.read_csv(CSV_FILE)

# The Paderborn dataset contains multiple "profile sessions" (test runs).
# We extract a single operational profile for stable identification.
if 'profile_id' in df.columns:
    profile_id_to_use = df['profile_id'].unique()[0] # Pick the first available profile
    df_profile = df[df['profile_id'] == profile_id_to_use].reset_index(drop=True)
else:
    df_profile = df

# For training stability, we take a 2000-sample window (a high-frequency slice)
window_size = 2000
df_slice = df_profile.iloc[1000:1000+window_size].copy()

# Note: The Kaggle dataset provides normalized values. 
# For this script, we treat them as raw state values for the PINN to learn from.
t_np = np.linspace(0, 0.1, window_size) # Simulated 0.1s window (approx 20kHz sample rate)
id_np = df_slice['i_d'].values
iq_np = df_slice['i_q'].values
ud_np = df_slice['u_d'].values
uq_np = df_slice['u_q'].values
omega_np = df_slice['motor_speed'].values # Electrical speed

# Convert numpy arrays to PyTorch Tensors
t_data = torch.tensor(t_np, dtype=torch.float32).view(-1, 1).requires_grad_(True)
id_data = torch.tensor(id_np, dtype=torch.float32).view(-1, 1)
iq_data = torch.tensor(iq_np, dtype=torch.float32).view(-1, 1)
ud_data = torch.tensor(ud_np, dtype=torch.float32).view(-1, 1)
uq_data = torch.tensor(uq_np, dtype=torch.float32).view(-1, 1)
omega_el = torch.tensor(omega_np, dtype=torch.float32).view(-1, 1)

# ==========================================
# 2. THE PINN ARCHITECTURE
# ==========================================
class KaggleMotorPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # The Neural Network mapping Time -> (id, iq)
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2) 
        )
        
        # --- THE SYSTEM IDENTIFICATION PARAMETERS ---
        # Registered as learnable tensors. Starting with completely arbitrary guesses.
        self.Rs_guess = nn.Parameter(torch.tensor([0.100]))   # Guessing 100 mOhm
        self.Ld_guess = nn.Parameter(torch.tensor([0.00500])) # Guessing 5000 uH
        self.Lq_guess = nn.Parameter(torch.tensor([0.00500])) # Guessing 5000 uH
        self.psi_guess = nn.Parameter(torch.tensor([0.010]))  # Guessing 10 mVs

    def forward(self, t):
        currents = self.net(t)
        return currents[:, 0:1], currents[:, 1:2] # id_pred, iq_pred

# ==========================================
# 3. TRAINING LOOP
# ==========================================
model = KaggleMotorPINN()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

epochs = 3000
history = {'loss': [], 'Rs': [], 'Ld': [], 'Lq': [], 'psi': []}

print("🚀 Starting Inverse Identification on Real Data...")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. Neural Network Predictions
    id_pred, iq_pred = model(t_data)
    
    # 2. Data Loss (Fit to Kaggle Sensor Data)
    loss_data = torch.mean((id_pred - id_data)**2) + torch.mean((iq_pred - iq_data)**2)
    
    # 3. Automatic Differentiation for di/dt
    did_dt_pred = torch.autograd.grad(id_pred, t_data, grad_outputs=torch.ones_like(id_pred), create_graph=True)[0]
    diq_dt_pred = torch.autograd.grad(iq_pred, t_data, grad_outputs=torch.ones_like(iq_pred), create_graph=True)[0]
    
    # 4. Physics Loss (The PMSM Voltage Equations)
    # Using the Kaggle voltage measurements (ud, uq) as the forcing functions
    res_d = ud_data - (model.Rs_guess * id_pred + model.Ld_guess * did_dt_pred - omega_el * model.Lq_guess * iq_pred)
    res_q = uq_data - (model.Rs_guess * iq_pred + model.Lq_guess * diq_dt_pred + omega_el * (model.Ld_guess * id_pred + model.psi_guess))
    
    loss_phys = torch.mean(res_d**2) + torch.mean(res_q**2)
    
    # Total Loss (Weighted to balance real-world data magnitudes)
    loss = loss_data + (0.001 * loss_phys) 
    loss.backward()
    optimizer.step()
    
    # Record history
    if epoch % 50 == 0:
        history['loss'].append(loss.item())
        history['Rs'].append(model.Rs_guess.item())
        history['Ld'].append(model.Ld_guess.item())
        history['Lq'].append(model.Lq_guess.item())
        history['psi'].append(model.psi_guess.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Rs: {model.Rs_guess.item():.4f} | Ld: {model.Ld_guess.item():.5f}")

# ==========================================
# 4. DASHBOARD VISUALIZATION
# ==========================================
# Print Final Discovered Parameters
print("\n--- Final Normalized Discovered Parameters ---")
print(f"Rs:  {model.Rs_guess.item():.5f}")
print(f"Ld:  {model.Ld_guess.item():.6f}")
print(f"Lq:  {model.Lq_guess.item():.6f}")
print(f"Psi: {model.psi_guess.item():.5f}")

print("\n✅ Training Complete. Generating Dashboard...")

plt.figure(figsize=(14, 10))
plt.suptitle("PINN System Identification Dashboard (Paderborn LEA Motor Data)", fontsize=16)

# Plot 1: Sensor Tracking
plt.subplot(2, 2, 1)
with torch.no_grad():
    id_final, iq_final = model(t_data)
plt.plot(t_data.detach().numpy(), id_data.numpy(), 'k--', label="Kaggle True $i_d$", alpha=0.6)
plt.plot(t_data.detach().numpy(), id_final.numpy(), 'b-', label="PINN Predicted $i_d$")
plt.title("Sensor Tracking ($i_d$)")
plt.xlabel("Time Window")
plt.ylabel("Normalized Current")
plt.legend()
plt.grid(True)

# Plot 2: Convergence of Resistance
plt.subplot(2, 2, 2)
plt.plot(history['Rs'], color='blue', linewidth=2)
plt.title("Convergence of Stator Resistance ($R_s$)")
plt.xlabel("Logging Step (x50 Epochs)")
plt.ylabel("Value")
plt.grid(True)

# Plot 3: Convergence of Inductances
plt.subplot(2, 2, 3)
plt.plot(history['Ld'], color='green', label='Discovered $L_d$')
plt.plot(history['Lq'], color='orange', label='Discovered $L_q$')
plt.title("Convergence of Inductances ($L_d, L_q$)")
plt.xlabel("Logging Step (x50 Epochs)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# Plot 4: Total Loss
plt.subplot(2, 2, 4)
plt.plot(history['loss'], color='red')
plt.yscale('log')
plt.title("Total Loss ($L_{data} + L_{phys}$)")
plt.xlabel("Logging Step (x50 Epochs)")
plt.ylabel("Loss (Log Scale)")
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


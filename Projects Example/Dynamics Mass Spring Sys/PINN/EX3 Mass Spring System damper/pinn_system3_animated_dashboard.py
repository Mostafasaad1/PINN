import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ==========================================
# 1. PHYSICAL PARAMETERS & MODEL ARCHITECTURE
# ==========================================
m, k, c, g = 10.0, 200.0, 15.0, 9.81
t_max = 5.0
MODEL_PATH = "Projects Example/Dynamics Mass Spring Sys/PINN/EX3 Mass Spring System damper/system3_pinn_model.pth"

class VibrationPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 3) # Outputs: [x, y, lambda]
        )
    def forward(self, t): return self.net(t)

# ==========================================
# 2. LOAD MODEL & GENERATE TRAJECTORIES
# ==========================================
pinn = VibrationPINN()
if os.path.exists(MODEL_PATH):
    pinn.load_state_dict(torch.load(MODEL_PATH))
    pinn.eval()
    print("--- Loaded Pre-Trained PINN Digital Twin ---")
else:
    raise FileNotFoundError(f"Model {MODEL_PATH} not found. Please run the training script first.")

# Generate Data via Inference
num_frames = 250
t_vals = np.linspace(0, t_max, num_frames)
t_tensor = torch.tensor(t_vals, dtype=torch.float32, requires_grad=True).view(-1, 1)

# Query the PINN
pred = pinn(t_tensor)
x_hist = pred[:, 0].detach().numpy()
lam_hist = pred[:, 2].detach().numpy()

# Use Automatic Differentiation to get Velocity for the Phase Portrait
x_tensor = pred[:, 0:1]
v_tensor = torch.autograd.grad(x_tensor, t_tensor, torch.ones_like(x_tensor))[0]
v_hist = v_tensor.detach().numpy().flatten()

# ==========================================
# 3. DRAWING HELPERS (EMULATION)
# ==========================================
def get_spring_coords(x_end, num_coils=10, width=0.15):
    x = np.linspace(0, x_end, num_coils * 2 + 1)
    y = np.zeros_like(x)
    y[1:-1:2] = width
    y[2:-1:2] = -width
    return x, y

# ==========================================
# 4. DASHBOARD SETUP (4-PANEL GRID)
# ==========================================
fig = plt.figure(figsize=(14, 10))
fig.suptitle("PINN Digital Twin: Animated Vibration Diagnostics Dashboard", fontsize=16)

# Grid Layout
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# Panel 1: Physical Emulation (Top Left)
ax_emu = fig.add_subplot(gs[0, 0])
ax_emu.set_xlim(-0.5, 2.5)
ax_emu.set_ylim(-1, 1)
ax_emu.set_aspect('equal')
ax_emu.set_title("Mechanical Emulation")
ax_emu.axvline(0, color='black', lw=4) # Wall
mass_rect = plt.Rectangle((0, -0.25), 0.5, 0.5, fc='dodgerblue', ec='black')
ax_emu.add_patch(mass_rect)
spring_line, = ax_emu.plot([], [], 'g-', lw=1.5, label='Spring (k)')

# Damper (Dashpot) Lines
d_cyl_1, = ax_emu.plot([], [], 'k-', lw=2)
d_cyl_2, = ax_emu.plot([], [], 'k-', lw=2)
d_rod, = ax_emu.plot([], [], 'k-', lw=3)
d_plate, = ax_emu.plot([], [], 'k-', lw=4)

# Panel 2: Displacement Plot (Top Right)
ax_x = fig.add_subplot(gs[0, 1])
ax_x.set_xlim(0, t_max)
ax_x.set_ylim(min(x_hist)-0.2, max(x_hist)+0.2)
ax_x.set_title("Displacement x(t)")
ax_x.set_xlabel("Time [s]")
ax_x.set_ylabel("Position [m]")
ax_x.grid(True)
line_x, = ax_x.plot([], [], 'b-', lw=2)
point_x, = ax_x.plot([], [], 'ro')

# Panel 3: Phase Portrait (Bottom Left)
ax_phase = fig.add_subplot(gs[1, 0])
ax_phase.set_xlim(min(x_hist)-0.2, max(x_hist)+0.2)
ax_phase.set_ylim(min(v_hist)-1.0, max(v_hist)+1.0)
ax_phase.set_title("State-Space Phase Portrait")
ax_phase.set_xlabel("Displacement x [m]")
ax_phase.set_ylabel("Velocity v [m/s]")
ax_phase.grid(True)
line_phase, = ax_phase.plot([], [], 'purple', lw=2)
point_phase, = ax_phase.plot([], [], 'ro')

# Panel 4: Constraint Force (Bottom Right)
ax_lam = fig.add_subplot(gs[1, 1])
ax_lam.set_xlim(0, t_max)
ax_lam.set_ylim(min(lam_hist)-10, max(lam_hist)+10)
ax_lam.set_title("Identified Constraint Force λ(t)")
ax_lam.set_xlabel("Time [s]")
ax_lam.set_ylabel("Normal Force [N]")
ax_lam.axhline(-m*g, color='red', linestyle='--', label='Theoretical -mg')
ax_lam.grid(True)
ax_lam.legend()
line_lam, = ax_lam.plot([], [], 'orange', lw=2)
point_lam, = ax_lam.plot([], [], 'ro')

# ==========================================
# 5. ANIMATION LOGIC
# ==========================================
def init():
    # Reset all dynamic elements
    mass_rect.set_xy((0, -0.25))
    spring_line.set_data([], [])
    for d_line in [d_cyl_1, d_cyl_2, d_rod, d_plate, line_x, point_x, line_phase, point_phase, line_lam, point_lam]:
        d_line.set_data([], [])
    return [mass_rect, spring_line, d_cyl_1, d_cyl_2, d_rod, d_plate, line_x, point_x, line_phase, point_phase, line_lam, point_lam]

def update(frame):
    t, x, v, lam = t_vals[frame], x_hist[frame], v_hist[frame], lam_hist[frame]
    
    # 1. Emulation Update
    mass_rect.set_xy((x, -0.25))
    sx, sy = get_spring_coords(x, width=0.1)
    spring_line.set_data(sx, sy + 0.15)
    
    h, off = 0.1, -0.15
    d_cyl_1.set_data([0, x*0.6], [h+off, h+off])
    d_cyl_2.set_data([0, x*0.6], [-h+off, -h+off])
    d_rod.set_data([x*0.4+0.1, x], [off, off])
    d_plate.set_data([x*0.4+0.1, x*0.4+0.1], [h*0.7+off, -h*0.7+off])
    
    # 2. Displacement Update (Draw trail up to current frame)
    line_x.set_data(t_vals[:frame], x_hist[:frame])
    point_x.set_data([t], [x])
    
    # 3. Phase Portrait Update
    line_phase.set_data(x_hist[:frame], v_hist[:frame])
    point_phase.set_data([x], [v])
    
    # 4. Constraint Force Update
    line_lam.set_data(t_vals[:frame], lam_hist[:frame])
    point_lam.set_data([t], [lam])
    
    return [mass_rect, spring_line, d_cyl_1, d_cyl_2, d_rod, d_plate, line_x, point_x, line_phase, point_phase, line_lam, point_lam]

# Create Animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=20)

plt.tight_layout()
# Save the final compiled frame as a static image
plt.savefig("system3_dashboard_final_frame.png", dpi=150)
print("Animation playing. Final frame saved as system3_dashboard_final_frame.png")

# Optional: To save as a video, uncomment the line below (requires ffmpeg installed)
# ani.save("system3_pinn_dashboard.mp4", writer='ffmpeg', fps=30)

plt.show()
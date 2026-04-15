# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        THE 3D PAPER AIRPLANE: A Physics-Informed Neural Network in 3D       ║
# ║                    Explained for 7-year-olds (and curious adults)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE STORY ──────────────────────────────────────────────────────────────────
#
#  Nettie is at the park. She throws a paper airplane. Now, the airplane moves 
#  in all THREE spatial dimensions at the same time:
#    → X: Forward (The direction she threw it)
#    → Y: Sideways (A strong wind is blowing from the left to the right)
#    ↑ Z: Up/Down (Gravity pulling it to the grass)
#
#  Nettie's job: Guess the exact (X, Y, Z) coordinate of the plane at ANY time.
#
#  Nettie's Teachers:
#
#   📺  THE VIDEO TEACHER  (→ "Data Loss")
#       Nettie has a 3D camera, but it's glitchy. She only gets 20 frames of 
#       video for the entire flight.
#
#   👨  DAD  (→ "Physics Loss")
#       Dad is back, and in 3D, he has THREE Unbreakable Rules:
#       1. The X-Rule (Forward): No motor. Forward speed never speeds up. 
#          (Math: d²x/dt² = 0)
#       2. The Y-Rule (Sideways Wind): A steady wind PUSHES the plane right.
#          (Math: d²y/dt² = 2.0 m/s²)  <-- Acceleration from the wind!
#       3. The Z-Rule (Gravity): Gravity always pulls down.
#          (Math: d²z/dt² = −9.81 m/s²)
#
#       If Nettie's guesses break ANY of the three rules → PHYSICS LOSS.
#
# ── HOW TO RUN ─────────────────────────────────────────────────────────────────
#   pip install torch numpy matplotlib
#   python pinn_3d_airplane.py
#
# ══════════════════════════════════════════════════════════════════════════════

import torch          
import torch.nn as nn 
import numpy as np    
import math           
import matplotlib.pyplot as plt 
# We need this specific import for 3D plotting in matplotlib!
from mpl_toolkits.mplot3d import Axes3D 

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SET UP THE 3D PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════

g = 9.81         # Gravity (Z-axis)
a_wind = 2.0     # Wind acceleration pushing right (Y-axis)

# The throw!
z0 = 1.5         # Thrown from 1.5 meters high (Nettie's shoulder)
v_x = 4.0        # Thrown FORWARD at 4 m/s
v_y = 0.0        # Initially not moving sideways (wind takes over later)
v_z = 5.0        # Thrown slightly UPWARD at 5 m/s

# How long until it hits the grass (Z = 0)?
t_end = (v_z + math.sqrt(v_z**2 + 2.0 * g * z0)) / g  # ≈ 1.26 seconds

device = torch.device('cpu')

print("=" * 65)
print("  ✈️   THE 3D PAPER AIRPLANE: Physics-Informed Neural Network")
print("=" * 65)
print(f"\n🕐  The plane is flying for {t_end:.3f} seconds.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: THE REAL ANSWER (The Ground Truth)
# ══════════════════════════════════════════════════════════════════════════════

def true_solution_3d(t):
    """Returns the true X, Y, and Z positions."""
    x_true = v_x * t
    y_true = 0.5 * a_wind * t**2              # Wind pushes it sideways!
    z_true = z0 + v_z * t - 0.5 * g * t**2    # Gravity pulls it down!
    return x_true, y_true, z_true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MAKE THE TRAINING DATA (The 3D Camera)
# ══════════════════════════════════════════════════════════════════════════════

num_snapshots = 20
t_data = torch.linspace(0, t_end, num_snapshots, device=device).unsqueeze(1)

# Get the clean answers
x_c, y_c, z_c = true_solution_3d(t_data)

# Add glitchy camera noise to X, Y, and Z
torch.manual_seed(42)
x_data = x_c + 0.1 * torch.randn_like(x_c)
y_data = y_c + 0.1 * torch.randn_like(y_c)
z_data = z_c + 0.1 * torch.randn_like(z_c)

# Stack them! Shape becomes [20, 3] -> Columns are X, Y, Z.
xyz_data = torch.cat([x_data, y_data, z_data], dim=1)

print(f"📸  Created {num_snapshots} glitchy 3D video frames.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: THE PHYSICS CHECK POINTS
# ══════════════════════════════════════════════════════════════════════════════

num_physics_pts = 100
t_physics = (
    torch.linspace(0, t_end, num_physics_pts, device=device)
    .unsqueeze(1)
    .requires_grad_(True) 
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BUILD NETTIE'S 3D BRAIN  
# ══════════════════════════════════════════════════════════════════════════════

class Nettie3DGuesser(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Sequential(
            nn.Linear(1, 64), # A slightly bigger brain for 3D!
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3), # Output is THREE: (X, Y, Z)
        )

    def forward(self, t):
        return self.brain(t)

nettie = Nettie3DGuesser().to(device)
print(f"\n🧠  Nettie's 3D Brain initialized. Ready to learn X, Y, and Z.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: THE TEACHING TOOLS
# ══════════════════════════════════════════════════════════════════════════════

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(nettie.parameters(), lr=0.001) # Slightly slower learning rate
num_epochs = 8000

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: THE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  🌪️   TRAINING BEGINS IN 3D...")
print("=" * 65 + "\n")

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    # ── A. DATA LOSS ─────────────────────────────────────────────────────────
    xyz_pred_data = nettie(t_data)
    loss_data = loss_fn(xyz_pred_data, xyz_data)

    # ── B. PHYSICS LOSS (Dad's 3 Rules) ──────────────────────────────────────
    xyz_pred_phys = nettie(t_physics)
    
    # Split the 3 columns into X, Y, and Z
    x_pred = xyz_pred_phys[:, 0:1] 
    y_pred = xyz_pred_phys[:, 1:2]
    z_pred = xyz_pred_phys[:, 2:3]

    # --- Rule 1: X-Axis (Forward Motion) ---
    dx_dt = torch.autograd.grad(x_pred, t_physics, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t_physics, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
    loss_phys_x = loss_fn(d2x_dt2, torch.zeros_like(d2x_dt2)) # Target = 0

    # --- Rule 2: Y-Axis (Wind blowing sideways) ---
    dy_dt = torch.autograd.grad(y_pred, t_physics, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_physics, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]
    loss_phys_y = loss_fn(d2y_dt2, torch.full_like(d2y_dt2, a_wind)) # Target = 2.0 (Wind)

    # --- Rule 3: Z-Axis (Gravity pulling down) ---
    dz_dt = torch.autograd.grad(z_pred, t_physics, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]
    d2z_dt2 = torch.autograd.grad(dz_dt, t_physics, grad_outputs=torch.ones_like(dz_dt), create_graph=True)[0]
    loss_phys_z = loss_fn(d2z_dt2, torch.full_like(d2z_dt2, -g)) # Target = -9.81 (Gravity)

    # ── C. TOTAL LOSS ────────────────────────────────────────────────────────
    total_loss = loss_data + loss_phys_x + loss_phys_y + loss_phys_z

    total_loss.backward()
    optimizer.step()

    # ── D. THE STORY ─────────────────────────────────────────────────────────
    if epoch % 2000 == 0 or epoch == 1:
        print(f"📅  Epoch {epoch:4d}")
        print(f"   📺 Video Error:  {loss_data.item():.6f}")
        print(f"   👨 X-Rule (0):   {loss_phys_x.item():.6f}")
        print(f"   👨 Y-Rule (Wind):{loss_phys_y.item():.6f}")
        print(f"   👨 Z-Rule (Grav):{loss_phys_z.item():.6f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: DRAWING THE 3D FLIGHT PATH
# ══════════════════════════════════════════════════════════════════════════════

print("📈  Drawing the 3D Flight Path...")

t_plot = np.linspace(0, t_end, 300)
t_plot_tensor = torch.tensor(t_plot, dtype=torch.float32, device=device).unsqueeze(1)

nettie.eval()
with torch.no_grad():
    xyz_nettie = nettie(t_plot_tensor).numpy()
    x_nettie, y_nettie, z_nettie = xyz_nettie[:, 0], xyz_nettie[:, 1], xyz_nettie[:, 2]

x_real, y_real, z_real = true_solution_3d(t_plot)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d') # MAGIC 3D PLOTTER!

# Plot True Physics Path
ax.plot(x_real, y_real, z_real, "b-", linewidth=3, label="📐 True Physics Path")

# Plot Nettie's AI Guess
ax.plot(x_nettie, y_nettie, z_nettie, "r--", linewidth=2.5, label="🤖 Nettie's 3D Guess")

# Plot the Glitchy Camera Data
ax.scatter(
    x_data.numpy(), y_data.numpy(), z_data.numpy(), 
    color="black", s=50, marker="o", label="📸 20 Glitchy Video Frames"
)

# Set labels for all 3 dimensions
ax.set_xlabel('X: Forward Throw (m)', fontsize=10, labelpad=10)
ax.set_ylabel('Y: Sideways Wind (m)', fontsize=10, labelpad=10)
ax.set_zlabel('Z: Height (m)', fontsize=10, labelpad=10)

ax.set_title("3D PINN: Paper Airplane in a Windstorm", fontsize=16, fontweight="bold")
ax.legend(fontsize=12)

# Make the viewing angle nice
ax.view_init(elev=20, azim=-45) 

plt.tight_layout()
plt.savefig("pinn_3d_results.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: YOU ARE NOW A HERO
# ══════════════════════════════════════════════════════════════════════════════
print("""
╔════════════════════════════════════════════════════════════════════╗
║                   🦸‍♂️  YOU ARE NOW A HERO  🦸‍♂️                       ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  You have successfully gone from 1D, to 2D, to full 3D spatial     ║
║  modeling using Physics-Informed Neural Networks.                  ║
║                                                                    ║
║  Why does this matter for real-world AI?                           ║
║  Because the real world operates in 3D.                            ║
║                                                                    ║
║  If you want an AI to fly a drone, operate a 6-DOF industrial      ║
║  robot arm, or predict the flow of fluid through a 3D pipe, you    ║
║  cannot rely on data alone. Data is noisy (like our camera).       ║
║                                                                    ║
║  By defining the X, Y, and Z physical rules (Navier-Stokes,        ║
║  Kinematics, Thermodynamics) and forcing the AI to obey them       ║
║  using PyTorch's automatic differentiation, you create an AI       ║
║  that understands the universe it lives in.                        ║
╚════════════════════════════════════════════════════════════════════╝
""")
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        THE 2D ROBOT TOSS: A Physics-Informed Neural Network in 2D           ║
# ║                    Explained for 7-year-olds (and curious adults)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE STORY ──────────────────────────────────────────────────────────────────
#
#  Nettie built a little DIY robot arm in her room. The robot tosses a toy 
#  block across the room. Now, the block moves in TWO directions at once:
#    → X: Forward across the floor (Horizontal)
#    ↑ Y: Up and down in the air (Vertical)
#
#  Nettie's job: Guess the exact (X, Y) coordinate of the block at ANY time.
#
#  Nettie still has her TWO teachers:
#
#   📺  THE VIDEO TEACHER  (→ "Data Loss")
#       Nettie watches a blurry video of the block flying.
#       She only gets 15 frames of video. 
#       If her (X, Y) guess doesn't match the video frames → DATA LOSS.
#
#   👨  DAD  (→ "Physics Loss")
#       Dad enforces the Unbreakable Rules. But in 2D, there are TWO rules:
#       1. The X-Rule (Forward): Once the block leaves the robot, nothing pushes 
#          it forward or backward. Forward speed NEVER changes. 
#          (Math: d²x/dt² = 0)
#       2. The Y-Rule (Vertical): Gravity ALWAYS pulls it down at 9.8 m/s².
#          (Math: d²y/dt² = −g)
#
#       If Nettie's guesses break EITHER rule → PHYSICS LOSS.
#
# ── THE PHYSICS (Simply) ───────────────────────────────────────────────────────
#
#   Horizontal (X): x(t) = v_x * t             (Constant speed forward)
#   Vertical   (Y): y(t) = y₀ + v_y*t − ½·g·t² (Gravity pulling down)
#
# ── HOW TO RUN ─────────────────────────────────────────────────────────────────
#   pip install torch numpy matplotlib
#   python pinn_2d_trajectory.py
#
# ══════════════════════════════════════════════════════════════════════════════

import torch          
import torch.nn as nn 
import numpy as np    
import math           
import matplotlib.pyplot as plt 

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SET UP THE 2D PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════

g = 9.81   # Gravity

# The DIY Robot Arm tosses the block!
y0 = 0.5   # Robot releases the block 0.5 meters off the ground
v_x = 3.0  # Block moves FORWARD at 3 m/s
v_y = 6.0  # Block is thrown UPWARD at 6 m/s

# How long until it hits the floor (y = 0)?
# 0 = 0.5 + 6.0*t - 0.5*9.81*t^2
t_end = (v_y + math.sqrt(v_y**2 + 2.0 * g * y0)) / g  # ≈ 1.3 seconds

device = torch.device('cpu')

print("=" * 65)
print("  🤖  THE 2D ROBOT TOSS: Physics-Informed Neural Network")
print("=" * 65)
print(f"\n🕐  The block is in the air for {t_end:.3f} seconds.")
print(f"📐  Release height: {y0}m | Speed X: {v_x}m/s | Speed Y: {v_y}m/s\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: THE REAL ANSWER (The Ground Truth)
# ══════════════════════════════════════════════════════════════════════════════

def true_solution(t):
    """
    Returns BOTH the true X and true Y positions for a given time.
    """
    x_true = v_x * t
    y_true = y0 + v_y * t - 0.5 * g * t**2
    return x_true, y_true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MAKE THE TRAINING DATA  (The Blurry Video)
# ══════════════════════════════════════════════════════════════════════════════

# 15 frames of video for the whole flight.
num_snapshots = 15
t_data = torch.linspace(0, t_end, num_snapshots, device=device).unsqueeze(1)

# Get the clean answers
x_clean, y_clean = true_solution(t_data)

# Add a tiny bit of blur/noise to BOTH X and Y
torch.manual_seed(42)
x_data = x_clean + 0.05 * torch.randn_like(x_clean)
y_data = y_clean + 0.05 * torch.randn_like(y_clean)

# Stack them together into one tensor! Shape becomes [15, 2]
# Column 0 is X data, Column 1 is Y data.
xy_data = torch.cat([x_data, y_data], dim=1)

print(f"📸  Created {num_snapshots} blurry video snapshots (containing X and Y).")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: THE PHYSICS CHECK POINTS
# ══════════════════════════════════════════════════════════════════════════════

num_physics_pts = 60

# We still only need speedometers on TIME, because we want to know how 
# X and Y change as TIME moves forward.
t_physics = (
    torch.linspace(0, t_end, num_physics_pts, device=device)
    .unsqueeze(1)
    .requires_grad_(True)  # ← MAGIC SPEEDOMETER ON TIME
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BUILD NETTIE'S 2D BRAIN  
# ══════════════════════════════════════════════════════════════════════════════

class Nettie2DGuesser(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Sequential(
            # Input is 1 (time)
            nn.Linear(1, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            # Output is TWO (X and Y coordinates)
            nn.Linear(40, 2), 
        )

    def forward(self, t):
        return self.brain(t)

nettie = Nettie2DGuesser().to(device)
print(f"\n🧠  Nettie is ready to guess in 2D! She has {sum(p.numel() for p in nettie.parameters())} knobs to tune.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: THE TEACHING TOOLS
# ══════════════════════════════════════════════════════════════════════════════

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(nettie.parameters(), lr=0.002)

num_epochs = 6000
history = {"total": [], "data": [], "physics_x": [], "physics_y": []}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: THE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  🎮  TRAINING BEGINS IN 2D...")
print("=" * 65 + "\n")

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    # ── B. DATA LOSS (The Video) ─────────────────────────────────────────────
    
    # Nettie outputs a block of guesses: [15 rows, 2 columns (X and Y)]
    xy_pred_data = nettie(t_data)
    
    # Compare Nettie's [X,Y] guesses to the blurry video [X,Y] data
    loss_data = loss_fn(xy_pred_data, xy_data)

    # ── C. PHYSICS LOSS (Dad's 2 Rules) ──────────────────────────────────────
    
    xy_pred_phys = nettie(t_physics)
    
    # We must split the output to check the rules independently!
    # Column 0 is the X guesses, Column 1 is the Y guesses.
    x_pred = xy_pred_phys[:, 0:1] 
    y_pred = xy_pred_phys[:, 1:2]

    # --- Rule 1: The X-Axis (Forward Motion) ---
    # Velocity X (dx/dt)
    dx_dt = torch.autograd.grad(
        x_pred, t_physics, grad_outputs=torch.ones_like(x_pred), create_graph=True
    )[0]
    
    # Acceleration X (d²x/dt²)
    d2x_dt2 = torch.autograd.grad(
        dx_dt, t_physics, grad_outputs=torch.ones_like(dx_dt), create_graph=True
    )[0]
    
    # Dad's Rule for X: Acceleration must be ZERO (constant forward speed)
    loss_physics_x = loss_fn(d2x_dt2, torch.zeros_like(d2x_dt2))


    # --- Rule 2: The Y-Axis (Gravity) ---
    # Velocity Y (dy/dt)
    dy_dt = torch.autograd.grad(
        y_pred, t_physics, grad_outputs=torch.ones_like(y_pred), create_graph=True
    )[0]
    
    # Acceleration Y (d²y/dt²)
    d2y_dt2 = torch.autograd.grad(
        dy_dt, t_physics, grad_outputs=torch.ones_like(dy_dt), create_graph=True
    )[0]
    
    # Dad's Rule for Y: Acceleration must be -9.81
    target_gravity = torch.full_like(d2y_dt2, -g)
    loss_physics_y = loss_fn(d2y_dt2, target_gravity)

    # ── D. TOTAL LOSS ────────────────────────────────────────────────────────
    
    # Add everything together. 
    # Notice we give Dad two votes now (one for X, one for Y).
    total_loss = loss_data + loss_physics_x + loss_physics_y

    history["total"].append(total_loss.item())
    history["data"].append(loss_data.item())
    history["physics_x"].append(loss_physics_x.item())
    history["physics_y"].append(loss_physics_y.item())

    # ── E & F. BACKPROP AND STEP ─────────────────────────────────────────────
    total_loss.backward()
    optimizer.step()

    # ── G. THE STORY ─────────────────────────────────────────────────────────
    if epoch % 1000 == 0 or epoch == 1:
        print(f"📅  Epoch {epoch:5d}")
        print(f"   📺 Video scolding:        {loss_data.item():.6f}")
        print(f"   👨 Dad's X-Rule scolding: {loss_physics_x.item():.6f}")
        print(f"   👨 Dad's Y-Rule scolding: {loss_physics_y.item():.6f}")
        print(f"   📊 Total scolding:        {total_loss.item():.6f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PRETTY PLOTS IN 2D
# ══════════════════════════════════════════════════════════════════════════════

print("📈  Drawing the 2D Trajectory...")

# We don't just want Height vs Time anymore. 
# We want to see Height (Y) vs Distance (X)!

t_plot = np.linspace(0, t_end, 200)
t_plot_tensor = torch.tensor(t_plot, dtype=torch.float32, device=device).unsqueeze(1)

nettie.eval()
with torch.no_grad():
    xy_nettie = nettie(t_plot_tensor).numpy()
    x_nettie = xy_nettie[:, 0]
    y_nettie = xy_nettie[:, 1]

x_real, y_real = true_solution(t_plot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("2D PINN: Predicting a Robot's Throw Trajectory", fontsize=16, fontweight="bold")

# ── Left Plot: The Actual 2D Flight Path (Y vs X) ─────────────────────────────
ax1.plot(x_real, y_real, "b-", linewidth=3, label="📐 Real Physics Path")
ax1.plot(x_nettie, y_nettie, "r--", linewidth=2.5, label="🤖 Nettie's 2D Guess")

# Plot the 15 blurry training data points (X vs Y)
ax1.scatter(
    x_data.numpy().flatten(), y_data.numpy().flatten(),
    color="black", zorder=5, s=70, marker="x", linewidths=2,
    label="📸 15 Blurry Video Frames"
)

# Draw the robot arm origin
ax1.scatter(0, y0, color="green", s=150, zorder=6, label="🦾 Robot Arm Release Point")

ax1.set_xlabel("Distance Across Floor (X metres)", fontsize=12)
ax1.set_ylabel("Height in Air (Y metres)", fontsize=12)
ax1.set_title("The Trajectory Plane", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)
ax1.axhline(0, color="saddlebrown", linewidth=2)
ax1.set_xlim(-0.5, max(x_real)*1.1)

# ── Right Plot: Loss History ──────────────────────────────────────────────────
epochs_x = list(range(1, num_epochs + 1))
ax2.semilogy(epochs_x, history["total"], "k-", linewidth=2, label="📊 Total Loss")
ax2.semilogy(epochs_x, history["data"], "b--", alpha=0.7, label="📺 Data Loss")
ax2.semilogy(epochs_x, history["physics_x"], "g:", alpha=0.7, label="👨 Physics X (Forward)")
ax2.semilogy(epochs_x, history["physics_y"], "r:", alpha=0.7, label="👨 Physics Y (Gravity)")

ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss (log scale)", fontsize=12)
ax2.set_title("Learning 2 Rules at Once", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("pinn_2d_results.png", dpi=140)
print("💾  Saved plot to: pinn_2d_results.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: THE 2D TAKEAWAY
# ══════════════════════════════════════════════════════════════════════════════
print("""
╔════════════════════════════════════════════════════════════════════╗
║                   🎓  THE 2D TAKEAWAY  🎓                          ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  You just scaled from a 1D problem to a Multi-Output problem!      ║
║                                                                    ║
║  Notice how PyTorch handles this easily:                           ║
║  1. The Neural Network outputs an array of variables: [X, Y].      ║
║  2. We slice the array to isolate the variables:                   ║
║     x_pred = xy_pred[:, 0:1]                                       ║
║  3. We apply different physics equations to different slices!      ║
║                                                                    ║
║  In real-world trajectory planning, this exact architecture is     ║
║  used. Instead of X and Y, the outputs could be Theta_1, Theta_2,  ║
║  and Theta_3 (the joint angles of an articulated arm), and the     ║
║  Physics Loss is the rigid body dynamics of the robot!             ║
╚════════════════════════════════════════════════════════════════════╝
""")
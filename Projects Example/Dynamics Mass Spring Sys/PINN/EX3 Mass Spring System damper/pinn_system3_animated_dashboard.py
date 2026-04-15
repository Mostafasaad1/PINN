"""
PINN Digital Twin: Animated Vibration Diagnostics Dashboard with Interactive Sliders

This script provides an animated visualization of the PINN-based digital twin 
for vibration analysis with interactive parameter adjustment.

Usage:
    python pinn_system3_animated_dashboard.py

Features:
    - Interactive sliders for all physical parameters
    - Real-time animation of mass-spring-damper system
    - Phase portrait and constraint force visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
import os

# ==========================================
# 1. DEFAULT PHYSICAL PARAMETERS
# ==========================================
DEFAULT_M = 10.0      # Mass [kg]
DEFAULT_K = 200.0     # Stiffness [N/m]
DEFAULT_C = 15.0      # Damping [Ns/m]
DEFAULT_G = 9.81      # Gravity [m/s^2]
DEFAULT_X0 = 1.2      # Initial position [m]
DEFAULT_V0 = 3.0      # Initial velocity [m/s]
DEFAULT_T_MAX = 5.0   # Simulation time [s]
DEFAULT_NUM_FRAMES = 250  # Animation frames

# ==========================================
# 2. PINN ARCHITECTURE
# ==========================================
class VibrationPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 3)  # Outputs: [x, y, lambda]
        )
    def forward(self, t): 
        return self.net(t)

# ==========================================
# 3. PHYSICS LOSS FUNCTION
# ==========================================
def get_physics_loss(model, t_collocation, m, k, c, g):
    """Physics loss function with parameterized physical constants."""
    t_collocation.requires_grad = True
    pred = model(t_collocation)
    x, y, lam = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    dx_dt = torch.autograd.grad(x, t_collocation, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t_collocation, torch.ones_like(dx_dt), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t_collocation, torch.ones_like(y), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_collocation, torch.ones_like(dy_dt), create_graph=True)[0]

    res_x = m * d2x_dt2 + c * dx_dt + k * x
    res_y = m * d2y_dt2 + lam + m * g
    res_phi = y
    return torch.mean(res_x**2) + torch.mean(res_y**2) + 10 * torch.mean(res_phi**2)

# ==========================================
# 4. TRAINING FUNCTION
# ==========================================
def train_pinn(m, k, c, g, x0, v0, t_max, epochs=12001, verbose=True):
    """Train a PINN model with given parameters."""
    pinn = VibrationPINN()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    t_train = torch.linspace(0, t_max, 600).view(-1, 1)
    
    if verbose:
        print(f"\n--- Training PINN ---")
        print(f"Parameters: m={m}kg, k={k}N/m, c={c}Ns/m, x0={x0}m, v0={v0}m/s")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_p = get_physics_loss(pinn, t_train, m, k, c, g)

        t0 = torch.tensor([[0.0]], requires_grad=True)
        p0 = pinn(t0)
        x0_p = p0[:, 0:1]
        v0_p = torch.autograd.grad(x0_p, t0, torch.ones_like(x0_p), create_graph=True)[0]
        loss_ic = (x0_p - x0)**2 + (v0_p - v0)**2 + (p0[:, 1:2])**2

        total_loss = loss_p + 50 * loss_ic
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 3000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {total_loss.item():.8f}")
    
    if verbose:
        print("Training complete!")
    return pinn

# ==========================================
# 5. NUMERICAL SOLUTION
# ==========================================
def get_numerical_solution(m, k, c, x0, v0, t_max, num_points=500):
    """Get numerical solution using RK45."""
    def mbd_ground_truth(t, state):
        x, v = state
        return [v, (-k*x - c*v) / m]
    
    t_eval = np.linspace(0, t_max, num_points)
    sol = solve_ivp(mbd_ground_truth, [0, t_max], [x0, v0], t_eval=t_eval, method='RK45')
    return t_eval, sol.y[0], sol.y[1]

# ==========================================
# 6. DRAWING HELPERS
# ==========================================
def get_spring_coords(x_end, num_coils=10, width=0.15):
    """Generate spring coordinates for visualization."""
    x = np.linspace(0, x_end, num_coils * 2 + 1)
    y = np.zeros_like(x)
    y[1:-1:2] = width
    y[2:-1:2] = -width
    return x, y

# ==========================================
# 7. INTERACTIVE DASHBOARD CLASS
# ==========================================
class InteractiveDashboard:
    def __init__(self):
        # Initialize parameters
        self.m = DEFAULT_M
        self.k = DEFAULT_K
        self.c = DEFAULT_C
        self.g = DEFAULT_G
        self.x0 = DEFAULT_X0
        self.v0 = DEFAULT_V0
        self.t_max = DEFAULT_T_MAX
        self.num_frames = DEFAULT_NUM_FRAMES
        
        # Data storage
        self.pinn = None
        self.t_vals = None
        self.x_hist = None
        self.v_hist = None
        self.lam_hist = None
        
        # Setup figure and sliders
        self.setup_figure()
        self.setup_sliders()
        
    def setup_figure(self):
        """Setup the main figure with subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("PINN Digital Twin: Interactive Vibration Dashboard", fontsize=14)
        
        # Adjust layout to make room for sliders
        self.fig.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.92)
        
        # Grid Layout for plots
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
        
        # Panel 1: Physical Emulation (Top Left)
        self.ax_emu = self.fig.add_subplot(gs[0, 0])
        self.ax_emu.set_xlim(-0.5, 3.0)
        self.ax_emu.set_ylim(-1, 1)
        self.ax_emu.set_aspect('equal')
        self.ax_emu.set_title("Mechanical Emulation")
        self.ax_emu.axvline(0, color='black', lw=4)  # Wall
        self.mass_rect = plt.Rectangle((0, -0.25), 0.5, 0.5, fc='dodgerblue', ec='black')
        self.ax_emu.add_patch(self.mass_rect)
        self.spring_line, = self.ax_emu.plot([], [], 'g-', lw=1.5, label='Spring (k)')
        
        # Damper (Dashpot) Lines
        self.d_cyl_1, = self.ax_emu.plot([], [], 'k-', lw=2)
        self.d_cyl_2, = self.ax_emu.plot([], [], 'k-', lw=2)
        self.d_rod, = self.ax_emu.plot([], [], 'k-', lw=3)
        self.d_plate, = self.ax_emu.plot([], [], 'k-', lw=4)
        
        # Panel 2: Displacement Plot (Top Right)
        self.ax_x = self.fig.add_subplot(gs[0, 1])
        self.ax_x.set_title("Displacement x(t)")
        self.ax_x.set_xlabel("Time [s]")
        self.ax_x.set_ylabel("Position [m]")
        self.ax_x.grid(True)
        self.line_x, = self.ax_x.plot([], [], 'b-', lw=2)
        self.point_x, = self.ax_x.plot([], [], 'ro')
        
        # Panel 3: Phase Portrait (Bottom Left)
        self.ax_phase = self.fig.add_subplot(gs[1, 0])
        self.ax_phase.set_title("State-Space Phase Portrait")
        self.ax_phase.set_xlabel("Displacement x [m]")
        self.ax_phase.set_ylabel("Velocity v [m/s]")
        self.ax_phase.grid(True)
        self.line_phase, = self.ax_phase.plot([], [], 'purple', lw=2)
        self.point_phase, = self.ax_phase.plot([], [], 'ro')
        
        # Panel 4: Constraint Force (Bottom Right)
        self.ax_lam = self.fig.add_subplot(gs[1, 1])
        self.ax_lam.set_title("Identified Constraint Force λ(t)")
        self.ax_lam.set_xlabel("Time [s]")
        self.ax_lam.set_ylabel("Normal Force [N]")
        self.ax_lam.grid(True)
        self.line_lam, = self.ax_lam.plot([], [], 'orange', lw=2)
        self.point_lam, = self.ax_lam.plot([], [], 'ro')
        self.theory_line = None
        
        # Info text
        self.info_text = self.fig.text(0.02, 0.98, '', fontsize=10, verticalalignment='top',
                                        fontfamily='monospace')
        
    def setup_sliders(self):
        """Setup interactive sliders."""
        slider_color = 'lightblue'
        
        # Create slider axes
        ax_m = self.fig.add_axes([0.15, 0.25, 0.3, 0.02])
        ax_k = self.fig.add_axes([0.15, 0.21, 0.3, 0.02])
        ax_c = self.fig.add_axes([0.15, 0.17, 0.3, 0.02])
        ax_g = self.fig.add_axes([0.15, 0.13, 0.3, 0.02])
        ax_x0 = self.fig.add_axes([0.6, 0.25, 0.3, 0.02])
        ax_v0 = self.fig.add_axes([0.6, 0.21, 0.3, 0.02])
        ax_tmax = self.fig.add_axes([0.6, 0.17, 0.3, 0.02])
        
        # Create sliders
        self.slider_m = Slider(ax_m, 'Mass [kg]', 1.0, 50.0, valinit=self.m, color=slider_color)
        self.slider_k = Slider(ax_k, 'Stiffness [N/m]', 50.0, 1000.0, valinit=self.k, color=slider_color)
        self.slider_c = Slider(ax_c, 'Damping [Ns/m]', 1.0, 100.0, valinit=self.c, color=slider_color)
        self.slider_g = Slider(ax_g, 'Gravity [m/s²]', 0.0, 20.0, valinit=self.g, color=slider_color)
        self.slider_x0 = Slider(ax_x0, 'Init Pos [m]', -5.0, 5.0, valinit=self.x0, color=slider_color)
        self.slider_v0 = Slider(ax_v0, 'Init Vel [m/s]', -10.0, 10.0, valinit=self.v0, color=slider_color)
        self.slider_tmax = Slider(ax_tmax, 'Time [s]', 1.0, 20.0, valinit=self.t_max, color=slider_color)
        
        # Connect sliders to update function
        self.slider_m.on_changed(self.update_params)
        self.slider_k.on_changed(self.update_params)
        self.slider_c.on_changed(self.update_params)
        self.slider_g.on_changed(self.update_params)
        self.slider_x0.on_changed(self.update_params)
        self.slider_v0.on_changed(self.update_params)
        self.slider_tmax.on_changed(self.update_params)
        
        # Train button
        ax_button = self.fig.add_axes([0.6, 0.08, 0.15, 0.04])
        self.train_button = Button(ax_button, 'Train PINN', color='lightgreen', hovercolor='green')
        self.train_button.on_clicked(self.train_and_animate)
        
        # Quick preview button
        ax_preview = self.fig.add_axes([0.8, 0.08, 0.15, 0.04])
        self.preview_button = Button(ax_preview, 'Quick Preview', color='lightyellow', hovercolor='yellow')
        self.preview_button.on_clicked(self.quick_preview)
        
    def update_params(self, val):
        """Update parameters from sliders."""
        self.m = self.slider_m.val
        self.k = self.slider_k.val
        self.c = self.slider_c.val
        self.g = self.slider_g.val
        self.x0 = self.slider_x0.val
        self.v0 = self.slider_v0.val
        self.t_max = self.slider_tmax.val
        
        # Update info text
        wn = np.sqrt(self.k/self.m)
        zeta = self.c / (2 * np.sqrt(self.m*self.k))
        damping_type = "Underdamped" if zeta < 1 else ("Critically Damped" if zeta == 1 else "Overdamped")
        
        info = f"System Properties:\n"
        info += f"  ωn = {wn:.2f} rad/s ({wn/(2*np.pi):.2f} Hz)\n"
        info += f"  ζ = {zeta:.3f} ({damping_type})"
        self.info_text.set_text(info)
        
    def train_and_animate(self, event=None):
        """Train PINN and run animation."""
        # Train PINN
        self.pinn = train_pinn(self.m, self.k, self.c, self.g, self.x0, self.v0, self.t_max)
        
        # Generate data
        self.t_vals = np.linspace(0, self.t_max, self.num_frames)
        t_tensor = torch.tensor(self.t_vals, dtype=torch.float32, requires_grad=True).view(-1, 1)
        
        # Query PINN
        pred = self.pinn(t_tensor)
        self.x_hist = pred[:, 0].detach().numpy()
        self.lam_hist = pred[:, 2].detach().numpy()
        
        # Get velocity
        x_tensor = pred[:, 0:1]
        v_tensor = torch.autograd.grad(x_tensor, t_tensor, torch.ones_like(x_tensor))[0]
        self.v_hist = v_tensor.detach().numpy().flatten()
        
        # Update axis limits
        self.ax_emu.set_xlim(-0.5, max(self.x_hist)+1.5)
        self.ax_x.set_xlim(0, self.t_max)
        self.ax_x.set_ylim(min(self.x_hist)-0.2, max(self.x_hist)+0.2)
        self.ax_phase.set_xlim(min(self.x_hist)-0.2, max(self.x_hist)+0.2)
        self.ax_phase.set_ylim(min(self.v_hist)-1.0, max(self.v_hist)+1.0)
        self.ax_lam.set_xlim(0, self.t_max)
        self.ax_lam.set_ylim(min(self.lam_hist)-10, max(self.lam_hist)+10)
        
        # Update theoretical line
        if self.theory_line:
            self.theory_line.remove()
        self.theory_line = self.ax_lam.axhline(-self.m*self.g, color='red', linestyle='--', 
                                                 label=f'Theoretical -mg = {-self.m*self.g:.1f} N')
        self.ax_lam.legend()
        
        # Run animation
        self.run_animation()
        
    def quick_preview(self, event=None):
        """Quick preview using numerical solution only."""
        # Get numerical solution
        t_eval, x_num, v_num = get_numerical_solution(self.m, self.k, self.c, 
                                                        self.x0, self.v0, self.t_max)
        
        # Store for animation
        self.t_vals = t_eval
        self.x_hist = x_num
        self.v_hist = v_num
        self.lam_hist = np.full_like(t_eval, -self.m * self.g)  # Approximate
        
        # Update axis limits
        self.ax_emu.set_xlim(-0.5, max(self.x_hist)+1.5)
        self.ax_x.set_xlim(0, self.t_max)
        self.ax_x.set_ylim(min(self.x_hist)-0.2, max(self.x_hist)+0.2)
        self.ax_phase.set_xlim(min(self.x_hist)-0.2, max(self.x_hist)+0.2)
        self.ax_phase.set_ylim(min(self.v_hist)-1.0, max(self.v_hist)+1.0)
        self.ax_lam.set_xlim(0, self.t_max)
        self.ax_lam.set_ylim(-self.m*self.g-20, -self.m*self.g+20)
        
        # Update theoretical line
        if self.theory_line:
            self.theory_line.remove()
        self.theory_line = self.ax_lam.axhline(-self.m*self.g, color='red', linestyle='--',
                                                 label=f'Theoretical -mg = {-self.m*self.g:.1f} N')
        self.ax_lam.legend()
        
        # Run animation
        self.run_animation()
        
    def init_animation(self):
        """Initialize animation."""
        self.mass_rect.set_xy((0, -0.25))
        self.spring_line.set_data([], [])
        for d_line in [self.d_cyl_1, self.d_cyl_2, self.d_rod, self.d_plate, 
                       self.line_x, self.point_x, self.line_phase, self.point_phase, 
                       self.line_lam, self.point_lam]:
            d_line.set_data([], [])
        return [self.mass_rect, self.spring_line, self.d_cyl_1, self.d_cyl_2, 
                self.d_rod, self.d_plate, self.line_x, self.point_x, 
                self.line_phase, self.point_phase, self.line_lam, self.point_lam]
    
    def update_animation(self, frame):
        """Update animation frame."""
        t, x, v, lam = self.t_vals[frame], self.x_hist[frame], self.v_hist[frame], self.lam_hist[frame]
        
        # 1. Emulation Update
        self.mass_rect.set_xy((x, -0.25))
        sx, sy = get_spring_coords(x, width=0.1)
        self.spring_line.set_data(sx, sy + 0.15)
        
        h, off = 0.1, -0.15
        self.d_cyl_1.set_data([0, x*0.6], [h+off, h+off])
        self.d_cyl_2.set_data([0, x*0.6], [-h+off, -h+off])
        self.d_rod.set_data([x*0.4+0.1, x], [off, off])
        self.d_plate.set_data([x*0.4+0.1, x*0.4+0.1], [h*0.7+off, -h*0.7+off])
        
        # 2. Displacement Update
        self.line_x.set_data(self.t_vals[:frame], self.x_hist[:frame])
        self.point_x.set_data([t], [x])
        
        # 3. Phase Portrait Update
        self.line_phase.set_data(self.x_hist[:frame], self.v_hist[:frame])
        self.point_phase.set_data([x], [v])
        
        # 4. Constraint Force Update
        self.line_lam.set_data(self.t_vals[:frame], self.lam_hist[:frame])
        self.point_lam.set_data([t], [lam])
        
        return [self.mass_rect, self.spring_line, self.d_cyl_1, self.d_cyl_2, 
                self.d_rod, self.d_plate, self.line_x, self.point_x, 
                self.line_phase, self.point_phase, self.line_lam, self.point_lam]
    
    def run_animation(self):
        """Run the animation."""
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=len(self.t_vals),
                                  init_func=self.init_animation, blit=True, interval=20)
        self.fig.canvas.draw()
        
    def show(self):
        """Display the dashboard."""
        plt.show()

# ==========================================
# 8. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("PINN Digital Twin: Interactive Vibration Dashboard")
    print("="*60)
    print("\nInstructions:")
    print("  1. Adjust sliders to set physical parameters")
    print("  2. Click 'Train PINN' to train and animate")
    print("  3. Click 'Quick Preview' for numerical solution only")
    print("="*60)
    
    # Create and show dashboard
    dashboard = InteractiveDashboard()
    dashboard.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# --- 1. Physics Engine ---
def get_trajectory(m, k, c, y_init, v_init, t_max=10, dt=0.02):
    """Calculates the physics over a set time using Euler integration."""
    t = np.arange(0, t_max, dt)
    y = np.zeros_like(t)
    v = np.zeros_like(t)
    y[0] = y_init
    v[0] = v_init
    
    # Calculate motion: m*a = -k*y - c*v
    for i in range(1, len(t)):
        acc = -(k/m)*y[i-1] - (c/m)*v[i-1]
        v[i] = v[i-1] + acc * dt
        y[i] = y[i-1] + v[i] * dt
        
    return t, y

# Initial parameters
m_init, k_init, c_init = 1.0, 20.0, 1.0
# Start the mass pulled down by 1.5 meters
t_data, y_data = get_trajectory(m_init, k_init, c_init, y_init=-1.5, v_init=0.0)

# --- 2. Visual Emulation Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.35) # Make room for sliders at the bottom

# Set the visible bounds of the "room"
ax.set_xlim(-2, 2)
ax.set_ylim(-3, 0.5)
ax.set_aspect('equal')
ax.set_title("Live Mass-Spring-Damper Emulation", fontsize=14)
ax.axis('off') # Hide standard graph axes for a clean look

# Draw the ceiling
ax.plot([-1.5, 1.5], [0, 0], color='black', lw=4)

# Create the visual elements (empty for now, updated in animation loop)
spring_line, = ax.plot([], [], color='gray', lw=2)
mass_box = plt.Rectangle((-0.4, -0.4), 0.8, 0.4, color='royalblue', ec='black', lw=2)
ax.add_patch(mass_box)

def get_spring_coords(y_mass, width=0.4, num_coils=10):
    """Generates x, y coordinates for a zig-zag spring that stretches."""
    y_points = np.linspace(0, y_mass, num_coils * 2 + 2)
    x_points = np.zeros_like(y_points)
    
    # Create alternating zig-zag pattern
    for i in range(1, len(x_points) - 1):
        if i % 2 == 0:
            x_points[i] = -width / 2
        else:
            x_points[i] = width / 2
            
    return x_points, y_points

# --- 3. Animation Loop ---
def update(frame):
    # Loop the animation data seamlessly
    idx = frame % len(t_data)
    current_y = y_data[idx]
    
    # Update mass position (offset by the box's height so it hangs from the top)
    mass_box.set_xy((-0.4, current_y - 0.4)) 
    
    # Update spring stretch
    xs, ys = get_spring_coords(current_y)
    spring_line.set_data(xs, ys)
    
    return spring_line, mass_box

# --- 4. Interactive UI Controls ---
axcolor = 'lightgray'
# [left, bottom, width, height]
ax_m = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_k = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_c = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

s_m = Slider(ax_m, 'Mass (m)', 0.1, 5.0, valinit=m_init)
s_k = Slider(ax_k, 'Stiffness (k)', 5.0, 50.0, valinit=k_init)
s_c = Slider(ax_c, 'Damping (c)', 0.0, 10.0, valinit=c_init)

def on_slider_update(val):
    global t_data, y_data
    # Recalculate the entire physics trajectory with the new slider values
    # We reset the initial position to -1.5 so you can see the new drop response
    t_data, y_data = get_trajectory(s_m.val, s_k.val, s_c.val, y_init=-1.5, v_init=0.0)

# Connect sliders to the update function
s_m.on_changed(on_slider_update)
s_k.on_changed(on_slider_update)
s_c.on_changed(on_slider_update)

# --- 5. Drop Button ---
resetax = plt.axes([0.85, 0.10, 0.1, 0.13])
button = Button(resetax, 'Drop\nMass', color='lightblue', hovercolor='skyblue')

def drop_mass(event):
    # Triggers a recalculation from a stretched state to restart the animation
    on_slider_update(None)
    
button.on_clicked(drop_mass)

# Start the animation engine (interval is in milliseconds per frame)
ani = FuncAnimation(fig, update, frames=len(t_data), interval=20, blit=True)

plt.show()
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define System Parameters
m = 1.0   # Mass (kg)
k = 20.0  # Spring constant (N/m)
c = 1.0   # Damping coefficient (N*s/m)

# 2. Define the ODE system
def msd_system(t, y):
    """
    y[0] is position (x)
    y[1] is velocity (v)
    """
    x, v = y
    dxdt = v
    dvdt = -(k/m)*x - (c/m)*v
    return [dxdt, dvdt]

# 3. Setup Initial Conditions and Time Span
# Initial position = 1.0 meters, Initial velocity = 0.0 m/s
y0 = [1.0, 0.0] 
t_span = (0, 10)  # Simulate for 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500) # Time points to evaluate

# 4. Solve the ODE
solution = solve_ivp(msd_system, t_span, y0, t_eval=t_eval)

# 5. Plot the Results
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='Displacement (m)', color='b', linewidth=2)
plt.axhline(0, color='black', linestyle='--') # Equilibrium line

plt.title('Mass-Spring-Damper System Response')
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement (meters)')
plt.legend()
plt.grid(True)
plt.show()
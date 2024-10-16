import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81  
L = 1.0  
theta_0 = np.radians(120)  
omega_0 = 0.0  
dt = 0.05  
t_max = 20 
m = 1.0 

t = np.arange(0, t_max, dt)

theta = np.zeros_like(t)
omega = np.zeros_like(t)

theta[0] = theta_0
omega[0] = omega_0

def derivatives(state):
    theta, omega = state
    alpha = -(g / L) * np.sin(theta)  
    return np.array([omega, alpha])

def runge_kutta_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

for i in range(1, len(t)):
    state = np.array([theta[i-1], omega[i-1]])
    new_state = runge_kutta_step(state, dt)
    
    theta[i], omega[i] = new_state

fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('Simple Pendulum Simulation (Lagrangian, Runge-Kutta)')
ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')

line, = ax.plot([], [], 'o-', lw=2, color='blue', markersize=10)  

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = L * np.sin(theta[frame])
    y = -L * np.cos(theta[frame])
    line.set_data([0, x], [0, y])
    line.set_marker('o')
    line.set_markersize(15) 
    return line,

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50)

plt.show()

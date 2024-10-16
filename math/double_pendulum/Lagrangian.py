import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81 
L1 = 1.0  
L2 = 1.0  
m1 = 1.0  
m2 = 1.0  
theta1_0 = np.radians(120)  
theta2_0 = np.radians(-100)  
omega1_0 = 0.0  
omega2_0 = 0.0  
dt = 0.05  
t_max = 20 

t = np.arange(0, t_max, dt)

theta1 = np.zeros_like(t)
theta2 = np.zeros_like(t)
omega1 = np.zeros_like(t)
omega2 = np.zeros_like(t)

theta1[0] = theta1_0
theta2[0] = theta2_0
omega1[0] = omega1_0
omega2[0] = omega2_0

def derivatives(state):
    theta1, omega1, theta2, omega2 = state

    delta_theta = theta2 - theta1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta_theta)**2
    den2 = (L2 / L1) * den1

    alpha1 = ((m2 * L1 * omega1**2 * np.sin(delta_theta) * np.cos(delta_theta) +
               m2 * g * np.sin(theta2) * np.cos(delta_theta) +
               m2 * L2 * omega2**2 * np.sin(delta_theta) -
               (m1 + m2) * g * np.sin(theta1)) / den1)

    alpha2 = ((-L1 * omega1**2 * np.sin(delta_theta) -
               g * np.sin(theta2) +
               g * np.sin(theta1) * np.cos(delta_theta)) / den2)

    return np.array([omega1, alpha1, omega2, alpha2])

def runge_kutta_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

for i in range(1, len(t)):
    state = np.array([theta1[i-1], omega1[i-1], theta2[i-1], omega2[i-1]])
    new_state = runge_kutta_step(state, dt)
    
    theta1[i], omega1[i], theta2[i], omega2[i] = new_state

fig, ax = plt.subplots()
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect('equal')
ax.set_title('Double Pendulum Simulation (Lagrangian, Runge-Kutta)')
ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')

line1, = ax.plot([], [], 'o-', lw=2, color='blue')
line2, = ax.plot([], [], 'o-', lw=2, color='red')
trace, = ax.plot([], [], lw=1, color='green')  

trace_x = []
trace_y = []

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    trace.set_data([], [])  
    return line1, line2, trace

def update(frame):
    
    x1 = L1 * np.sin(theta1[frame])
    y1 = -L1 * np.cos(theta1[frame])
    x2 = x1 + L2 * np.sin(theta2[frame])
    y2 = y1 - L2 * np.cos(theta2[frame])
    
    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    
    trace_x.append(x2)
    trace_y.append(y2)
    trace.set_data(trace_x, trace_y)
    
    return line1, line2, trace

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50)

plt.show()

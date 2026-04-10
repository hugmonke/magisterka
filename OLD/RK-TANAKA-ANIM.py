import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
USE_RANDOM_PARAMS = False

param_ranges = {
    "alpha": (0, 300),
    "mu": (-0.5, 1.5),
    "omega": (0.05, 0.2),
    "beta": (0, 1),
    "gamma": (0, 1),
    "x0": (-1, 1),
    "y0": (-1, 1),
}

alpha = 222
mu    = 0.001
omega = 0.08
beta  = 0
gamma = 0
x0    = 0.1
y0    = 0.0

if USE_RANDOM_PARAMS:
    alpha = np.random.uniform(*param_ranges["alpha"])
    mu    = np.random.uniform(*param_ranges["mu"])
    omega = np.random.uniform(*param_ranges["omega"])
    beta  = np.random.uniform(*param_ranges["beta"])
    gamma = np.random.uniform(*param_ranges["gamma"])
    x0    = np.random.uniform(*param_ranges["x0"])
    y0    = np.random.uniform(*param_ranges["y0"])

print("Parameters:", alpha, mu, omega, beta, gamma, x0, y0)

# ------------------------------------------------------------
# Time setup
# ------------------------------------------------------------
t0 = 0.0
t_end = 200.0
dt = 0.01
N = int((t_end - t0) / dt)

t = np.zeros(N)
x = np.zeros(N)
y = np.zeros(N)

x[0] = x0
y[0] = y0

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def derivatives(x, y):
    dxdt = y
    dydt = alpha + mu*y - omega**2 * x - beta * x**2 - gamma * x**3
    return dxdt, dydt

# RK4
for i in range(N - 1):
    t[i+1] = t[i] + dt
    k1x, k1y = derivatives(x[i], y[i])
    k2x, k2y = derivatives(x[i] + 0.5*dt*k1x, y[i] + 0.5*dt*k1y)
    k3x, k3y = derivatives(x[i] + 0.5*dt*k2x, y[i] + 0.5*dt*k2y)
    k4x, k4y = derivatives(x[i] + dt*k3x, y[i] + dt*k3y)
    x[i+1] = x[i] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    y[i+1] = y[i] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)

# ------------------------------------------------------------
# FAST ANIMATION
# ------------------------------------------------------------
skip = 20   # Increase this to make animation faster

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

line1, = ax1.plot([], [], lw=1)
line2, = ax2.plot([], [], lw=1)
point, = ax2.plot([], [], 'ro', markersize=4)

ax1.set_xlim(0, t_end)
ax1.set_ylim(np.min(x), np.max(x))
ax1.set_title("x(t)")

ax2.set_xlim(np.min(x), np.max(x))
ax2.set_ylim(np.min(y), np.max(y))
ax2.set_title("Phase Space")


def update(frame):
    i = frame * skip
    if i >= N:
        i = N - 1

    line1.set_data(t[:i], x[:i])
    line2.set_data(x[:i], y[:i])
    point.set_data([x[i]], [y[i]])

    return line1, line2, point


frames = N // skip
ani = FuncAnimation(fig, update, frames=frames, interval=1, blit=True)

plt.tight_layout()
plt.show()
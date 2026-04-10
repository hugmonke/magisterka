import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
USE_RANDOM_PARAMS = False

param_ranges = {
    "alpha": (-1.0, 0.0),
    "mu": (-0.5, 1.5),
    "beta": (0.0, 1.0),
    "p": (0.0, 1.0),
    "q": (0.0, 1.5),
    "s": (0.0, 2.0),
    "x0": (-1, 1),
    "y0": (-1, 1),
    "z0": (-1, 1),
}

# Fixed parameters
alpha = -0.5
mu = 0.5
beta = 0.5
p = 0.2
q = 0.8
s = 1.0

x0 = 0.1
y0 = 0.0
z0 = 0.0

# Randomization
if USE_RANDOM_PARAMS:
    alpha = np.random.uniform(*param_ranges["alpha"])
    mu    = np.random.uniform(*param_ranges["mu"])
    beta  = np.random.uniform(*param_ranges["beta"])
    p     = np.random.uniform(*param_ranges["p"])
    q     = np.random.uniform(*param_ranges["q"])
    s     = np.random.uniform(*param_ranges["s"])
    x0    = np.random.uniform(*param_ranges["x0"])
    y0    = np.random.uniform(*param_ranges["y0"])
    z0    = np.random.uniform(*param_ranges["z0"])

print("Parameters:", alpha, mu, beta, p, q, s, x0, y0, z0)

# ------------------------------------------------------------
# TIME SETUP
# ------------------------------------------------------------
dt = 0.01
t_end = 200
N = int(t_end / dt)

t = np.zeros(N)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

x[0], y[0], z[0] = x0, y0, z0

# ------------------------------------------------------------
# DERIVATIVES
# ------------------------------------------------------------
def derivatives(x, y, z):
    dxdt = y
    dydt = alpha*x + mu*y + z
    dzdt = -beta*y - p*z - q*y + s*y*z
    return dxdt, dydt, dzdt

# ------------------------------------------------------------
# RK4 INTEGRATION
# ------------------------------------------------------------
for i in range(N - 1):
    k1x, k1y, k1z = derivatives(x[i], y[i], z[i])
    k2x, k2y, k2z = derivatives(x[i]+0.5*dt*k1x, y[i]+0.5*dt*k1y, z[i]+0.5*dt*k1z)
    k3x, k3y, k3z = derivatives(x[i]+0.5*dt*k2x, y[i]+0.5*dt*k2y, z[i]+0.5*dt*k2z)
    k4x, k4y, k4z = derivatives(x[i]+dt*k3x, y[i]+dt*k3y, z[i]+dt*k3z)

    x[i+1] = x[i] + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
    y[i+1] = y[i] + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
    z[i+1] = z[i] + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
    t[i+1] = t[i] + dt

# ------------------------------------------------------------
# FAST ANIMATION SETTINGS
# ------------------------------------------------------------
skip = 200   # increase → faster animation

# ------------------------------------------------------------
# FIGURE SETUP
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 5))

# Time plot
ax1 = fig.add_subplot(1, 2, 1)
line1, = ax1.plot([], [], lw=1)
ax1.set_xlim(0, t_end)
ax1.set_ylim(np.min(x), np.max(x))
ax1.set_title("x(t)")

# 3D phase space
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
line2, = ax2.plot([], [], [], lw=0.7)
point, = ax2.plot([], [], [], 'ro', markersize=4)

ax2.set_xlim(np.min(x), np.max(x))
ax2.set_ylim(np.min(y), np.max(y))
ax2.set_zlim(np.min(z), np.max(z))
ax2.set_title("3D Phase Space")

# ------------------------------------------------------------
# UPDATE FUNCTION
# ------------------------------------------------------------
def update(frame):
    i = frame * skip
    if i >= N:
        i = N - 1

    # time plot
    line1.set_data(t[:i], x[:i])

    # 3D trajectory
    line2.set_data(x[:i], y[:i])
    line2.set_3d_properties(z[:i])

    # moving point
    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])

    return line1, line2, point

# ------------------------------------------------------------
# RUN ANIMATION
# ------------------------------------------------------------
frames = N // skip
ani = FuncAnimation(fig, update, frames=frames, interval=1)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = -0.5
mu = 0.5
beta = 0.5
p = 3.2
q = 0.5
s = 1.0

# Time
dt = 0.01
t_end = 200
N = int(t_end / dt)

# Arrays
t = np.zeros(N)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

# Initial conditions
x[0] = 0.1
y[0] = 0.0
z[0] = 0.0

# Derivatives
def derivatives(x, y, z):
    dxdt = y
    dydt = alpha*x + mu*y + z
    dzdt = -beta*y - p*z - q*y + s*y*z
    return dxdt, dydt, dzdt

# RK4
for i in range(N-1):
    k1x, k1y, k1z = derivatives(x[i], y[i], z[i])
    k2x, k2y, k2z = derivatives(x[i]+0.5*dt*k1x, y[i]+0.5*dt*k1y, z[i]+0.5*dt*k1z)
    k3x, k3y, k3z = derivatives(x[i]+0.5*dt*k2x, y[i]+0.5*dt*k2y, z[i]+0.5*dt*k2z)
    k4x, k4y, k4z = derivatives(x[i]+dt*k3x, y[i]+dt*k3y, z[i]+dt*k3z)

    x[i+1] = x[i] + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
    y[i+1] = y[i] + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
    z[i+1] = z[i] + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
    t[i+1] = t[i] + dt

# Plot
plt.plot(t, x)
plt.title("Irregular Stellar Pulsation")
plt.xlabel("Time")
plt.ylabel("Radius")
plt.show()

# 3D phase space
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
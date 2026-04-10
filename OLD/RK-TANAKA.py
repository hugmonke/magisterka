import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Takeuti & Tanaka Nonlinear Stellar Pulsation Model
# ------------------------------------------------------------
# This model describes nonlinear radial pulsations of a star.
# It is a system of two coupled nonlinear ODEs:
#
#   dx/dt = y
#   dy/dt = α + μ*y − ω^2*x − β*x^2 − γ*x^3
#
# x = radial displacement
# y = velocity
#
# Parameters control driving, damping, and nonlinear effects.
# ------------------------------------------------------------

# Model parameters (you can experiment with these)
alpha = 222    # driving force
mu    = .001     # damping/driving coefficient
omega = 0.08    # natural pulsation frequency
beta  = 0    # quadratic nonlinearity
gamma = 0    # cubic nonlinearity

# Time setup
t0 = 0.0
t_end = 100.0
dt = 0.01
N = int((t_end - t0) / dt)

# Arrays to store solution
t = np.zeros(N)
x = np.zeros(N)
y = np.zeros(N)

# Initial conditions
x[0] = 0.1   # initial displacement
y[0] = 0   # initial velocity


# ------------------------------------------------------------
# Define the system of ODEs
# ------------------------------------------------------------
def derivatives(x, y):
    dxdt = y
    dydt = alpha + mu*y - omega**2 * x - beta * x**2 - gamma * x**3
    return dxdt, dydt


# ------------------------------------------------------------
# Runge-Kutta 4th Order Method
# ------------------------------------------------------------
for i in range(N - 1):
    t[i+1] = t[i] + dt

    k1x, k1y = derivatives(x[i], y[i])
    k2x, k2y = derivatives(x[i] + 0.5*dt*k1x, y[i] + 0.5*dt*k1y)
    k3x, k3y = derivatives(x[i] + 0.5*dt*k2x, y[i] + 0.5*dt*k2y)
    k4x, k4y = derivatives(x[i] + dt*k3x, y[i] + dt*k3y)

    x[i+1] = x[i] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    y[i+1] = y[i] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)

# ------------------------------------------------------------
# Poincare Map
# ------------------------------------------------------------
poincare_x = []

for i in range(1, N):
    if y[i-1] < 0 and y[i] >= 0:
        poincare_x.append(x[i])

# Build map x_n vs x_{n+1}
xn = poincare_x[:-1]
xn1 = poincare_x[1:]

plt.figure()
plt.scatter(xn, xn1, s=10)
plt.xlabel("x_n")
plt.ylabel("x_{n+1}")
plt.title("Poincaré Map")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
plt.figure()
plt.plot(t, x)
plt.title("Nonlinear Stellar Pulsation (Takeuti-Tanaka Model)")
plt.xlabel("Time")
plt.ylabel("Radial Displacement")
plt.grid(True)
plt.show()


# Phase space plot
plt.figure()
plt.plot(x, y)
plt.title("Phase Space")
plt.xlabel("Displacement x")
plt.ylabel("Velocity y")
plt.grid(True)
plt.show()
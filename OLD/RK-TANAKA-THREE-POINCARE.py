import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------


# ------------------------------------------------------------
# POINCARÉ SECTION PLANE: a*x + b*y + c*z + d = 0
# ------------------------------------------------------------
def generate_poincare_plane(x0, y0, z0, axis='z', offset_ratio=0.5):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    if axis == 'x':
        a, b, c = 1.0, 0.0, 0.0
        d = -x0
    elif axis == 'y':
        a, b, c = 0.0, 1.0, 0.0
        d = -y0
    elif axis == 'z':
        a, b, c = 0.0, 0.0, 1.0
        d = -z0
    else:
        raise ValueError(f"Axis must be 'x', 'y', or 'z', is {axis}")

    return a, b, c, d


# ------------------------------------------------------------
# 1D Oscillator with Nonlinear Nonadiabatic Terms
# ------------------------------------------------------------
alpha = -0.5; mu = 0.5; beta = 0.5
p = 3.2; q = 0.5; s = 1

# Time
dt = 0.01
t_end = 200
ITER_NUM = int(t_end / dt)

# Arrays
t = np.zeros(ITER_NUM)
x = np.zeros(ITER_NUM)
y = np.zeros(ITER_NUM)
z = np.zeros(ITER_NUM)

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

def shannon_entropy_2d(x_points, y_points, bins=50):
    # NORMALIZED 
    H, _, _ = np.histogram2d(x_points, y_points, bins=bins)
    P = H / np.sum(H)  # Normalize to probabilities
    P_nonzero = P[P > 0] # Remove zeros (avoid log(0))
    
    entropy = -np.sum(P_nonzero * np.log(P_nonzero))
    
    max_entropy = np.log(bins * bins)  # maximum possible
    return entropy / max_entropy


# RK4
for i in range(ITER_NUM-1):
    k1x, k1y, k1z = derivatives(x[i], y[i], z[i])
    k2x, k2y, k2z = derivatives(x[i]+0.5*dt*k1x, y[i]+0.5*dt*k1y, z[i]+0.5*dt*k1z)
    k3x, k3y, k3z = derivatives(x[i]+0.5*dt*k2x, y[i]+0.5*dt*k2y, z[i]+0.5*dt*k2z)
    k4x, k4y, k4z = derivatives(x[i]+dt*k3x, y[i]+dt*k3y, z[i]+dt*k3z)

    x[i+1] = x[i] + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
    y[i+1] = y[i] + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
    z[i+1] = z[i] + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
    t[i+1] = t[i] + dt



def plane(x, y, z):
    return a*x + b*y + c*z + d

poincare_x = []
poincare_y = []
poincare_z = []
a, b, c, d = generate_poincare_plane(x[0], y[0], z[0], axis='z', offset_ratio=0.5)
for i in range(1, ITER_NUM):
    if plane(x[i-1], y[i-1], z[i-1]) < 0 and plane(x[i], y[i], z[i]) >= 0:
        # One point to the "left", the other to the "right" - means the plane cuts through
        poincare_x.append(x[i])
        poincare_y.append(y[i])
        poincare_z.append(z[i])


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

# Time series
plt.figure()
plt.plot(t, x)
plt.title("Irregular Stellar Pulsation")
plt.xlabel("t - Time")
plt.ylabel("x(t) - Displacement")
plt.grid()
plt.show()

# Phase space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Phase Space")
plt.show()

# Poincare map
plt.figure()
plt.scatter(poincare_x, poincare_z, s=10)
plt.xlabel("x")
plt.ylabel("z")
entropy = shannon_entropy_2d(poincare_x, poincare_z, bins=50)
print(f"Shannon entropy (Poincare map): {entropy}")
plt.title(f"Poincaré Map with entropy = {entropy:.3f}")
plt.grid()
plt.show()

# # Phase space + Poincaré map (plane) - Full trajectory

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, linewidth=0.5, alpha=0.4)

# # Poincare points

# ax.scatter(poincare_x, poincare_y, poincare_z, s=20, alpha=0.6)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.set_title("Phase Space with Poincaré Section")
# plt.show()


# Phase space + Poincaré map (PLANE) + points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=0.5, alpha=0.4)
ax.scatter(poincare_x, poincare_y, poincare_z, s=20, alpha=0.8)
# Entropy



# DRAW THE PLANE: a*x + b*y + c*z + d = 0
grid_size = 50
plt.rcParams['toolbar'] = 'toolmanager'
x_range = np.linspace(min(x), max(x), grid_size)
y_range = np.linspace(min(y), max(y), grid_size)
z_range = np.linspace(min(z), max(z), grid_size)


if c != 0: # z = f(x,y)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (-a*X - b*Y - d) / c
    ax.plot_surface(X, Y, Z, alpha=0.2)
elif b != 0: # y = f(x,z)
    z_range = np.linspace(min(z), max(z), grid_size)
    X, Z = np.meshgrid(x_range, z_range)
    Y = (-a*X - c*Z - d) / b
    ax.plot_surface(X, Y, Z, alpha=0.2)
elif a != 0: # x = f(z,y)
    y_range = np.linspace(min(y), max(y), grid_size)
    Z, Y = np.meshgrid(z_range, y_range)
    X = (-b*Y - c*Z - d) / a
    ax.plot_surface(X, Y, Z, alpha=0.2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Phase Space with Poincare Plane")

plt.show()
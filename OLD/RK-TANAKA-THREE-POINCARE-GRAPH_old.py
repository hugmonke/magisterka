import numpy as np
import matplotlib.pyplot as plt
import common as com 
# fajne Parameters: {'alpha': -0.5502103593764673, 'mu': 0.6915052784410058, 'beta': 0.5784962839255214, 'p': 4.010755401427479, 'q': 0.5108874980348869, 's': 1.6773485833457182}
# ------------------------------------------------------------
# RANDOM PARAMETERS
# ------------------------------------------------------------
def model_parameteres():
    params = {
              "alpha": np.random.uniform(-2, 2)
              ,"mu": np.random.uniform(-1, 1)
              ,"gamma": np.random.uniform(0.1, 3)
              ,"p": np.random.uniform(0, 3)
              ,"s": np.random.uniform(0.5, 2)
            }
    
    return params


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
def derivatives(x, y, z, params):
    #         Jacobian
    # |0         1            0|
    # |alfa      mu           1|
    # |0     -(beta + q)     -p|
    # beta and q are lineary dependent -> we merge them into one parameter for
    dxdt = y
    dydt = params["alpha"]*x + params["mu"]*y + z
    dzdt = -params["gamma"]*y - params["p"]*z + params["s"]*y*z
    return dxdt, dydt, dzdt

# ------------------------------------------------------------
# RK4 SOLVER
# ------------------------------------------------------------
def runge_kutta(x, y, z, param, dt):
    k1x, k1y, k1z = derivatives(x, y, z, param)
    k2x, k2y, k2z = derivatives(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z, param)
    k3x, k3y, k3z = derivatives(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z, param)
    k4x, k4y, k4z = derivatives(x + dt*k3x, y + dt*k3y, z + dt*k3z, param)

    x_next = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    y_next = y + dt*(k1y + 2*k2y + 2*k3y + k4y)/6
    z_next = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    
    return x_next, y_next, z_next

def solve_system(params, dt=0.01, t_skip=50, t_end=150):

    # We dont care about the simulation until initial t_skip time
    N_skip = int(t_skip / dt)
    N_sim = int(t_end / dt)

    x_arr = np.zeros(N_sim)
    y_arr = np.zeros(N_sim)
    z_arr = np.zeros(N_sim)

    x0 = 0.1
    y0 = 0
    z0 = 0

    for _ in range(N_skip):
        x0, y0, z0 = runge_kutta(x0, y0, z0, params, dt)

    x_arr[0], y_arr[0], z_arr[0] = x0, y0, z0
    for i in range(N_sim-1):
        x_arr[i+1], y_arr[i+1], z_arr[i+1] = runge_kutta(x_arr[i], y_arr[i], z_arr[i], params, dt)

        if np.abs(x_arr[i+1]) > 1e6 or np.abs(y_arr[i+1]) > 1e6:
            raise ValueError("System Diverged")
    return x_arr, y_arr, z_arr
# ------------------------------------------------------------
# POINCARÉ MAP
# ------------------------------------------------------------
def generate_plane(point, normal):
    point = np.array(point)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # normalize

    a, b, c = normal
    d = -np.dot(normal, point)

    return a, b, c, d


def poincare_map(x, y, z, plane):
    a, b, c, d = plane
    def plane_eq(x, y, z): 
        return a*x + b*y + c*z + d
    poinc_x = []
    poinc_y = [] 
    poinc_z = []

    for i in range(1, len(x)):
        f_prev = plane_eq(x[i-1], y[i-1], z[i-1])
        f_curr = plane_eq(x[i], y[i], z[i])

        # count BOTH upward and downward crossings
        if (f_prev < 0 and f_curr >= 0) or (f_prev > 0 and f_curr <= 0):
            t_cross = abs(f_prev)/(abs(f_prev)+abs(f_curr))
            xi = x[i-1] + t_cross*(x[i]-x[i-1])
            yi = y[i-1] + t_cross*(y[i]-y[i-1])
            zi = z[i-1] + t_cross*(z[i]-z[i-1])
            poinc_x.append(xi)
            poinc_y.append(yi)
            poinc_z.append(zi)

    return np.array(poinc_x), np.array(poinc_y), np.array(poinc_z)


# ------------------------------------------------------------
# ENTROPY
# ------------------------------------------------------------
def shannon_entropy(poinc_x, poinc_y, bins=50, floor=10):

    if len(poinc_x) < floor: 
        # If trajectory crossed plane less than *floor* times, its probably stable
        return 0
    
    H, _, _ = np.histogram2d(poinc_x, poinc_y, bins=bins)
    P = H / np.sum(H)
    P = P[P > 0]
    return -np.sum(P * np.log(P)) / np.log(bins * bins)


# ------------------------------------------------------------
# LYAPUNOV EXPONENT
# ------------------------------------------------------------
def lyapunov_exponent(params, dt=0.01, t_skip=50, t_end=150):
    """Returns Largest Lyapunov Exponent (LLE)."""
    N_skip = int(t_skip / dt)
    N_sim = int(t_end / dt)

    epsilon = 1e-8
    x1 = 0.1
    y1 = 0
    z1 = 0

    for _ in range(N_skip):
        x1, y1, z1 = runge_kutta(x1, y1, z1, params, dt)

    x2 = x1 + epsilon
    y2 = y1
    z2 = z1
    d0 = epsilon
    sum_log = 0

    for _ in range(N_sim):
        x1, y1, z1 = runge_kutta(x1, y1, z1, params, dt)
        x2, y2, z2 = runge_kutta(x2, y2, z2, params, dt)

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if d == 0 or np.isnan(d) or np.isinf(d):
            continue

        sum_log += np.log(d/d0)
        scale = d0/d
        x2 = x1 + dx * scale 
        y2 = y1 + dy * scale 
        z2 = z1 + dz * scale

    return sum_log / (N_sim * dt)


# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------
def classify(entropy, lle):
    if lle > 0.01:
        return "CHAOTIC"
    elif entropy < 0.2 and lle < -0.01:
        return "STABLE"
    else:
        return "PERIODIC"

def plot_poincare_plane(ax, plane, x, y, z, alpha=0.2):
    
    a, b, c, d = plane

    grid_size = 50
    x_range = np.linspace(min(x), max(x), grid_size)
    y_range = np.linspace(min(y), max(y), grid_size)
    z_range = np.linspace(min(z), max(z), grid_size)

    if c != 0:  # z = f(x,y)
        X, Y = np.meshgrid(x_range, y_range)
        Z = (-a*X - b*Y - d) / c
    elif b != 0:  # y = f(x,z)
        X, Z = np.meshgrid(x_range, z_range)
        Y = (-a*X - c*Z - d) / b
    elif a != 0:  # x = f(y,z)
        Z, Y = np.meshgrid(z_range, y_range)
        X = (-b*Y - c*Z - d) / a
    else:
        raise ValueError("Invalid plane definition")

    ax.plot_surface(X, Y, Z, alpha=alpha)


# ------------------------------------------------------------
# MAIN RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    np.seterr(all='raise')
    t_end = 150
    success = False

    while not success:
        params = model_parameteres()
        print("Parameters:", params)
        try:
            lle = lyapunov_exponent(params)
            x, y, z = solve_system(params)
            mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)

            dx0, dy0, dz0 = derivatives(x[0], y[0], z[0], params)
            plane = generate_plane(point=(mean_x, mean_y, mean_z), normal=(dx0, dy0, dz0))
            poinc_x, poinc_y, poinc_z = poincare_map(x, y, z, plane)
            
            entropy = shannon_entropy(poinc_x, poinc_y)
            state = classify(entropy, lle)
            success = True

        except (OverflowError, ValueError, FloatingPointError):
            print(f"Overflow for these parameters: {params}")
            pass
        
    print(f"Final State: {state} | Entropy: {entropy:.3f} | LLE: {lle:.3f}")
    

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(10, 8))

    # Time Series
    ax1 = fig.add_subplot(221)
    ax1.plot(np.linspace(0, 150, len(x)), x)
    ax1.set_title("Time Series x(t)")
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    # Phase Space
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.plot(x, y, z, lw=0.5)
    ax2.set_title("Phase Space")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    # Phase Space + Plane
    ax3 = fig.add_subplot(222, projection='3d')
    ax3.plot(x, y, z, alpha=0.3, lw=0.5)
    ax3.scatter(poinc_x, poinc_y, poinc_z, s=5, color='red')
    plot_poincare_plane(ax2, plane, x, y, z)
    ax3.set_title("Phase Space + Plane")
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')

    # Poincare Map
    ax4 = fig.add_subplot(224)
    ax4.scatter(poinc_x, poinc_y, s=5)
    ax4.set_title("Poincare Map")
    
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------
# def entropy_heatmap():
#     alpha_vals = np.linspace(-1.5, 0.5, 20)
#     p_vals = np.linspace(1.0, 5.0, 20)

#     heatmap = np.zeros((len(alpha_vals), len(p_vals)))

#     for i, a in enumerate(alpha_vals):
#         for j, p in enumerate(p_vals):
#             params = model_parameteres()
#             params["alpha"] = a
#             params["p"] = p

#             _, x, y, z = runge_kutta(params, t_end=100)
#             plane = generate_plane(x[0], y[0], z[0])
#             poinc_x, poinc_z = poincare_map(x, y, z, plane)

#             heatmap[i,j] = shannon_entropy(poinc_x, poinc_z)


#     plt.figure()
#     plt.imshow(heatmap, origin='lower', aspect='auto',
#                extent=[p_vals[0], p_vals[-1], alpha_vals[0], alpha_vals[-1]])
#     plt.colorbar(label="Entropy")
#     plt.xlabel("p")
#     plt.ylabel("alpha")
#     plt.title("Entropy Heatmap")
#     plt.show()


# Run heatmap
# entropy_heatmap()
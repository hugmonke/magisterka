import numpy as np
import matplotlib.pyplot as plt
import COMMON as com 
import tomllib
# ------------------------------------------------------------
# POINCARÉ MAP
# ------------------------------------------------------------
def generate_plane(point, normal):
    point = np.array(point)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # normalize

    a, b, c = normal
    d = -np.dot(normal.T, point.T)

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

def main():
    with open("config.toml", "rb") as conf:
        config = tomllib.load(conf)

    INIT_XYZ    = config.get("INIT_XYZ", [0.1, 0.0, 0.0])
    DT          = config.get("DT", 0.01)
    T_SKIP      = config.get("T_SKIP", 100)
    T_END       = config.get("T_END", 1000)
    CUTOFF      = config.get("CUTOFF", 1e6)
    SIZE        = config.get("SIZE", 1)
    PARAMS_LIST = config.get("SAVED_PARAMS", [{}])
    for PARAMS in PARAMS_LIST:
        success = False
        while not success:
            params = com.get_parameteres(params=PARAMS, size=1)
            lle, valid_mask, x, y, z = com.solve_and_get_lle(init_xyz = INIT_XYZ
                                                            , params = params
                                                            , dt = DT
                                                            , t_skip = T_SKIP
                                                            , t_end = T_END
                                                            , size = SIZE
                                                            , cutoff= CUTOFF
                                                            )
            if not valid_mask[0]: print("System diverged. Retrying with new parameters."); continue

            params = {param: val[0] for param, val in params.items()}
            param_string = ", ".join([f"'{param}': {val}" for param, val in params.items()])
            print("Parameters:", param_string)
            alpha, mu = params["alpha"], params["mu"]
            gamma, p, s = params["gamma"], params["p"], params["s"]
            x, y, z, lle = x[:, 0], y[:, 0], z[:, 0], lle[0]
            mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
            dx0, dy0, dz0 = com.get_derivatives(x=x[0], y=y[0], z=z[0], alpha=alpha, mu=mu, gamma=gamma, p=p, s=s)
            plane = generate_plane(point=(mean_x, mean_y, mean_z), normal=(dx0, dy0, dz0))
            poinc_x, poinc_y, poinc_z = poincare_map(x=x, y=y, z=z, plane=plane)
            entropy = com.shannon_entropy(poinc_x=poinc_x, poinc_y=poinc_y, bins=50, floor=10)
            state = com.classify(entropy=entropy, lle=lle)
            success = True
            
        print(f"Classified State: {state} | Entropy: {entropy:.4f} | LLE: {lle:.4f}")
        with open("simulation_results.txt", "a", encoding="utf-8") as file:
            log_line = f"Classified State: {state:<14} | Actual State:          | Entropy: {entropy:.4f} | LLE: {lle:.4f} | Params: {{{param_string}}} | T_SKIP: {T_SKIP} | T_END: {T_END}\n"
            file.write(log_line)
        print("-> Results appended to simulation_results.txt")

        # ------------------------------------------------------------
        # PLOTS
        # ------------------------------------------------------------

        fig = plt.figure(figsize=(10, 8))

        # Time Series
        ax1 = fig.add_subplot(221)
        ax1.plot(np.linspace(0, T_END, len(x)), x)
        ax1.set_title("Time Series x(t)")
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')

        # Phase Space
        ax2 = fig.add_subplot(223, projection='3d')
        ax2.scatter(x[0], y[0], z[0], color='green', s=40, zorder=5, label='Start')
        ax2.legend()
        ax2.plot(x, y, z, lw=0.5)
        ax2.set_title("Phase Space")
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')

        # Phase Space + Plane
        ax3 = fig.add_subplot(222, projection='3d')
        ax3.plot(x, y, z, alpha=0.3, lw=0.5)
        ax3.scatter(poinc_x, poinc_y, poinc_z, s=5, color='red')
        ax3.scatter(x[0], y[0], z[0], color='green', s=40, zorder=5, label='Start')
        ax3.legend()
        plot_poincare_plane(ax3, plane, x, y, z)
        ax3.set_title("Phase Space + Plane")
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')

        # Poincare Map
        ax4 = fig.add_subplot(224)
        ax4.scatter(poinc_x, poinc_y, s=5)
        ax4.set_title("Poincare Map")
        
        plt.tight_layout()
        plt.show()
# ------------------------------------------------------------
# MAIN RUN
# ------------------------------------------------------------
if __name__ == "__main__": main()
    


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
#     plt.title("Entropy Heatmap")z
#     plt.show()


# Run heatmap
# entropy_heatmap()
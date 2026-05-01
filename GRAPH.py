import numpy as np
import matplotlib.pyplot as plt
import COMMON as com 
import tomllib


def plot_poincare_plane(ax, plane, x, y, z, alpha=0.2):
    """Visualizes a 2D Poincaré section plane within a 3D phase space plot.

    Calculates a meshgrid surface based on the plane equation coefficients.
    Dynamically determines which coordinate to solve for based on the nonzero coefficients to avoid division by zero. 
    Resulting surface is constrained to the bounds of the provided trajectory data.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): Plot where the surface will be drawn.
        plane (tuple): Plane equation coefficients.
        x (array-like): Trajectory X-coordinates, used to define grid boundaries.
        y (array-like): Trajectory Y-coordinates, used to define grid boundaries.
        z (array-like): Trajectory Z-coordinates, used to define grid boundaries.
        alpha (float, optional): Transparency level of the plane surface. Defaults to 0.2.

    Raises:
        ValueError: If all coefficients (a, b, c) are zero, rendering the plane is impossible.
    """
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
    """Orchestrates the visual analysis and classification of saved stellar models.

    1. Loads parameter sets from 'config.toml', 
    2. Reconstructs 3D trajectories, and 
    3. Performs a multi-stage classification. 
    4. Generates a four-panel plot of time series, 3D phase space, intersecting Poincaré plane, and 2D Poincaré map.
    """
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
        params = com.get_parameteres(params=PARAMS, size=1)
        lle, valid_mask, x, y, z = com.solve_and_get_lle(init_xyz = INIT_XYZ
                                                        , params = params
                                                        , dt = DT
                                                        , t_skip = T_SKIP
                                                        , t_end = T_END
                                                        , size = SIZE
                                                        , cutoff= CUTOFF
                                                        )
        
        if not valid_mask[0]: 
            print(f"System diverged for {PARAMS}. \nContinuing.")
            continue
        
        params = {param: val[0] for param, val in params.items()}
        param_string = ", ".join([f"'{param}': {val}" for param, val in params.items()])
        print("Parameters:", param_string)

        alpha, mu = params["alpha"], params["mu"]
        gamma, p, s = params["gamma"], params["p"], params["s"]
        x, y, z, lle = x[:, 0], y[:, 0], z[:, 0], lle[0]

        mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
        dx0, dy0, dz0 = com.get_derivatives(x=x[0], y=y[0], z=z[0], alpha=alpha, mu=mu, gamma=gamma, p=p, s=s)
        plane = com.generate_plane(point=(mean_x, mean_y, mean_z), normal=(dx0, dy0, dz0))
        poinc_x, poinc_y, poinc_z = com.poincare_map(x=x, y=y, z=z, plane=plane)

        entropy = com.shannon_entropy(poinc_x=poinc_x, poinc_y=poinc_y)
        state = com.classify(entropy=entropy, lle=lle)  
        
        tolerance = 0.05
        state, features = com.validate_state_and_features(x_array=x, dt=DT, state=state, tolerance=tolerance)
        R21, phi21 = features["R21"], features["phi21"]
        R31, phi31 = features["R31"], features["phi31"]
            
        print(f"Classified State: {state} | Entropy: {entropy:.4f} | LLE: {lle:.4f}")
        with open("simulation_results.txt", "a", encoding="utf-8") as file:
            log_line = f"Classified State: {state:<14} | Actual State:          | Entropy: {entropy:.4f} | LLE: {lle:.4f} | Params: {{{param_string}}} | T_SKIP: {T_SKIP} | T_END: {T_END}\n"
            file.write(log_line)
        print("-> Results appended to simulation_results.txt")


        fig = plt.figure(figsize=(10, 8))

        # Time Series
        ax1 = fig.add_subplot(221)
        ax1.plot(np.linspace(0, T_END, len(x)), x)
        ax1.set_title("Time Series x(t)")
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        ax1.set_xlim(T_END-500, T_END)
        ax1.invert_yaxis()
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

if __name__ == "__main__": main()

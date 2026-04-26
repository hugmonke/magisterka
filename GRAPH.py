import numpy as np
import matplotlib.pyplot as plt
import COMMON as com 
import tomllib


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
        
        if state in ["CHAOTIC", "QUASI_PERIODIC", "PERIODIC"]:
            # Double check if PERIODIC, not DIVERGENT (The wave doesnt run away)
            drift_threshold = 0.05 # Chosen arbitrally, works best, allows some numerical errors
            half_idx = len(x)//2 
            drift_ratio = abs(np.mean(x[:half_idx]) - np.mean(x[half_idx:])) / (np.max(x) - np.min(x) + 1e-9)
            amp_start = np.max(x[:half_idx]) - np.min(x[:half_idx])
            amp_end = np.max(x[half_idx:]) - np.min(x[half_idx:])
            amp_growth = abs(amp_end - amp_start) / (amp_start + 1e-9)
                    
            if drift_ratio >= 0.05 or amp_growth >= 0.05: 
                state = "DIVERGENT"
            elif state == "PERIODIC":
                features = com.get_fourier_features(x, dt=DT) 
                if features: 
                    R21, PHI21 = features["R21"], features["phi21"]
                    R31, PHI31 = features["R31"], features["phi31"]
                else: 
                    state = "DIVERGENT"
            
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
        ax1.set_xlim(T_END-200, T_END)
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

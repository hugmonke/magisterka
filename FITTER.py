import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.optimize import differential_evolution
import COMMON as com 
import tomllib



# ------------------------------------------------------------
# PHYSICS ENGINE & FEATURE EXTRACTION
# ------------------------------------------------------------
def get_trajectory(params, init_xyz=(0.1, 0.0, 0.0), dt=0.01, t_skip=100, t_end=500):
    """Runs a single simulation and returns the steady-state x, y, z arrays."""
    x, y, z = init_xyz
    
    steps_skip = int(t_skip / dt)
    for _ in range(steps_skip):
        x, y, z = com.runge_kutta(x, y, z, dt, **params)
        if np.isnan(x) or abs(x) > 1e6:
            return None, None, None 

    steps_record = int(t_end / dt)
    x_hist = np.zeros(steps_record)
    y_hist = np.zeros(steps_record)
    z_hist = np.zeros(steps_record)
    
    for i in range(steps_record):
        x, y, z = com.runge_kutta(x, y, z, dt, **params)
        if np.isnan(x) or abs(x) > 1e6:
            return None, None, None
            
        x_hist[i], y_hist[i], z_hist[i] = x, y, z
        
    return x_hist, y_hist, z_hist


# ------------------------------------------------------------
# THE COST FUNCTION (Fitness test)
# ------------------------------------------------------------
def cost_function(target_featuers, param_array, param_names, dt, t_skip, t_end):
    """Calculates how 'wrong' the parameters are, penalizing quasi-periodicity."""
    params = {name: val for name, val in zip(param_names, param_array)}
    
    x_array, y_array, z_array = get_trajectory(params, dt=dt, t_skip=t_skip, t_end=t_end)
    if x_array is None: return 1e6 # Penalty: Explosion
        
    # PERIODICITY CHECK
    z_mean = np.mean(z_array)
    crossed_up = (z_array[:-1] <= z_mean) & (z_array[1:] > z_mean)
    crossed_ind = np.where(crossed_up)[0]
    if len(crossed_ind) < 2: return 1e6 # Penalty: Not oscillating
    poinc_x = x_array[crossed_ind] # X-coords at crossing points
    amplitude_x = np.max(x_array) - np.min(x_array)
    periodicity_error = np.std(poinc_x) / (amplitude_x + 1e-6) # normalized variance of crossings
    
    features = com.get_fourier_features(x_array, dt)
    if features is None: return 1e6 # Penalty: Fixed point (Dead)
        
    error_R21 = (features["R21"] - target_featuers["R21"])**2
    diff_phi = abs(features["phi21"] - target_featuers["phi21"])
    diff_phi = min(diff_phi, 2 * np.pi - diff_phi)
    error_phi21 = diff_phi**2
    
    # TOTAL COST
    # Add a massive multiplier to periodicity_error
    # If standard deviation != ~0, it will destroy the fitness score
    total_cost = + 1000*periodicity_error + 10*error_R21 + error_phi21 
    
    return total_cost

# ------------------------------------------------------------
# MAIN RUN
# ------------------------------------------------------------
def main():
    try:
        with open("config.toml", "rb") as conf:
            config = tomllib.load(conf)
    except FileNotFoundError as e:
        print(f"{e}: config.toml not found.")
        exit()

    DT              = config.get("DT", 0.01)
    T_SKIP          = config.get("T_SKIP", 100)
    T_END           = config.get("T_END", 500)
    OGLE_TARGETS    = {
                        "R21": 0.545,
                        "phi21": 4.395
                      }
    print("=== STARFIT: Tanaka-Takeuti Inverse Problem ===")
    print(f"Targeting OGLE RRab: R21={OGLE_TARGETS['R21']}, phi21={OGLE_TARGETS['phi21']}")
    
    BOUNDS          = [
                        (0.1, 10.0),       # alpha
                        (0.1, 10.0),       # mu
                        (0.0, 5.0),        # gamma
                        (0.0, 5.0),        # p
                        (0.0, 5.0)         # s
                       ]
    
    param_arr = ['alpha', 'mu', 'gamma', 'p', 's']
    
    print("Starting Differential Evolution Optimizer\n")
    result = differential_evolution(func=cost_function
                                    , bounds=BOUNDS
                                    , args=(OGLE_TARGETS, param_arr, DT, T_SKIP, T_END)
                                    , strategy='rand1bin'
                                    , maxiter=50
                                    , popsize=100
                                    , tol=0.01
                                    , polish=False
                                    , disp=True
                                    , workers=-1
                                    , updating='deferred'     # Required when using parallel workers    
                                    )
    
    print(f"=== OPTIMIZATION FINISHED (Final Error: {result.fun:.5f}) ===")
    if result.fun < 0.01:
        print("> RESULT: GOOD FIT")
    elif result.fun < 0.2:
        print("> RESULT: MID FIT")
    else:
        print("> RESULT: BAD FIT")
        
    best_params = {name: val for name, val in zip(param_arr, result.x)}
    final_x, final_y, final_z = get_trajectory(best_params, dt=DT, t_skip=T_SKIP, t_end=T_END)
    final_features = com.get_fourier_features(final_x, dt=DT)
    
    print("=== Best Parameters Found ===\n")
    for k, v in best_params.items():
        print(f"{k:>6}: {v:.4f}")
    print("=== Simulation vs Target ===\n")
    print(f"R21:   Sim = {final_features['R21']}      | Target={OGLE_TARGETS['R21']}")
    print(f"phi21: Sim = {final_features['phi21']}    | Target={OGLE_TARGETS['phi21']}")

    with open("config.toml", "a", encoding="utf-8") as conf:
        conf.write(f"\n[[SAVED_PARAMS]] # State: OPTIMIZED_FIT_ERR_{result.fun}\n") 
        for p_key, p_val in best_params.items():
            conf.write(f"{p_key} = {p_val}\n")
        print('\n-> BEST FIT SAVED TO: config.toml')

if __name__ == "__main__": main()
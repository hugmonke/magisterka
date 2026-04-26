import numpy as np
from scipy.optimize import differential_evolution
import COMMON as com 
import tomllib



# ------------------------------------------------------------
# PHYSICS ENGINE & FEATURE EXTRACTION
# ------------------------------------------------------------
def get_trajectory(params, init_xyz=(0.1, 0.0, 0.0), dt=0.01, t_skip=100, t_end=500, cutoff=150):
    """Runs a single simulation and returns the steady-state x, y, z arrays."""

    x, y, z = init_xyz
    N_skip = int(t_skip / dt)
    N_sim = int(t_end / dt)

    for _ in range(N_skip):
        x, y, z = com.runge_kutta(x, y, z, dt, **params)
        if np.isnan(x) or abs(x) > cutoff:
            return None, None, None 
    x_arr, y_arr, z_arr = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)

    for i in range(N_sim):
        x, y, z = com.runge_kutta(x, y, z, dt, **params)
        if np.isnan(x) or abs(x) > cutoff:
            return None, None, None
            
        x_arr[i], y_arr[i], z_arr[i] = x, y, z
        
    return x_arr, y_arr, z_arr


# ------------------------------------------------------------
# THE COST FUNCTION (Fitness test)
# ------------------------------------------------------------
def cost_function(param_array, target_featuers, param_names, dt, t_skip, t_end):
    """Calculates how 'wrong' the parameters are, penalizing quasi-periodicity."""
    params = {name: val for name, val in zip(param_names, param_array)}
    
    x_array, y_array, z_array = get_trajectory(params, dt=dt, t_skip=t_skip, t_end=t_end)
    if x_array is None: return 1e6 

    z_mean = np.mean(z_array)
    has_crossed = (z_array[:-1] <= z_mean) & (z_array[1:] > z_mean)
    where_crossed = np.where(has_crossed)[0]
    if len(where_crossed) < 2: return 1e6 

    crossed_x = x_array[where_crossed]
    x_amplitude = np.max(x_array) - np.min(x_array)
    if x_amplitude < 0.5: 
        return 1e6
    periodicity_error = np.std(crossed_x) / (x_amplitude + 1e-9) 
    
    features = com.get_fourier_features(x_array, dt)
    if features is None: return 1e6 
        
    error_R21 = (features["R21"] - target_featuers["R21"])**2
    diff_phi = abs(features["phi21"] - target_featuers["phi21"])
    diff_phi = min(diff_phi, 2 * np.pi - diff_phi)
    error_phi21 = diff_phi**2

    error_R31 = 0
    error_phi31 = 0
    if "R31" in target_featuers and "phi31" in target_featuers:
        error_R31 = (features["R31"] - target_featuers["R31"])**2
        diff_phi31 = abs(features["phi31"] - target_featuers["phi31"])
        error_phi31 = (min(diff_phi31, 2 * np.pi - diff_phi31))**2
    
    # TOTAL COST: Notice the weights!
    # R values are tiny (0.1 - 0.5), Phases are large (0 - 6.28)
    # We multiply the R errors by 10 to make the algorithm care about them equally.
    total_cost = periodicity_error + 10*error_R21 + error_phi21 + 10*error_R31 + error_phi31
    
    return total_cost

def create_seeded_population(base_params, bounds, popsize=100, spread_fraction=0.05):
    """
    Creates an initial population centered around a known good guess.
    spread_fraction controls how tight the cluster is (0.05 = 5% of the bound range).
    """
    num_params = len(bounds)
    population = np.zeros((popsize, num_params))
    population[0] = base_params
    
    for i in range(1, popsize):
        for j in range(num_params):
            bound_min, bound_max = bounds[j]
            bound_range = bound_max - bound_min
            
            noise = np.random.normal(loc=0.0, scale=bound_range * spread_fraction)
            new_val = base_params[j] + noise
            
            population[i, j] = np.clip(new_val, bound_min, bound_max)
            
    return population



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
    T_SKIP          = config.get("T_SKIP", 500)
    T_END           = config.get("T_END", 1000)
    SPREAD          = config.get("SPREAD", 0.01)
    POPSIZE          = config.get("POPSIZE", 100)
    OGLE_TARGETS    = {
                        "R21": 0.447
                        ,"phi21": 4.738         
                        ,"R31": 0.206
                        ,"phi31": 3.168
                      }
    print("=== STARFIT: Tanaka-Takeuti Inverse Problem ===")
    print(f"Targeting OGLE RRab: R21={OGLE_TARGETS['R21']}, phi21={OGLE_TARGETS['phi21']}, R31={OGLE_TARGETS.get(('R31'), None)}, phi31={OGLE_TARGETS.get(('phi31'), None)}")
    
    BOUNDS          = [
                        (-2, 4),       # alpha
                        (-1, 5),       # mu
                        (0, 5),        # gamma
                        (0, 5),        # p
                        (0, 5)         # s
                       ]
    
    param_arr = ['alpha', 'mu', 'gamma', 'p', 's']
    
    print("Starting Differential Evolution Optimizer\n")

    PARAMS = config.get("SAVED_STAR_PARAMS", [{}])[0]
    print(f"Looking around parameters: {PARAMS}")
    MAP_MAKER_GUESS = [PARAMS[k] for k in param_arr]

    seeded_population = create_seeded_population(base_params=MAP_MAKER_GUESS, 
                                          bounds=BOUNDS, 
                                          popsize=POPSIZE, 
                                          spread_fraction=SPREAD)
    
    result = differential_evolution(func=cost_function
                                    , bounds=BOUNDS
                                    , args=(OGLE_TARGETS, param_arr, DT, T_SKIP, T_END)
                                    , strategy='rand1bin'
                                    , maxiter=150
                                    , popsize=POPSIZE
                                    , init=seeded_population
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
    final_x, final_y, final_z = get_trajectory(best_params, dt=DT, t_skip=T_SKIP, t_end=T_END, cutoff=150)
    if final_x is None:
        print("\n[!] WARNING: The optimizer's best result is dynamically unstable (exploded).")
        print("[!] It tried to cheat the cutoff. Run the optimizer again.")
        return
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
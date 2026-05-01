import numpy as np
from scipy.optimize import differential_evolution
import COMMON as com 
import tomllib


def get_trajectory(params, init_xyz=(0.1, 0.0, 0.0), dt=0.01, t_skip=100, t_end=500, cutoff=150):
    """Simulates the dynamical system and returns the trajectory after 't_skip' time.

    Integrates the system's differential equations using a Runge-Kutta method. 
    Runs the system for 't_skip', then records the trajectory for the specified 't_end' duration.
    Aborts simulation if trajectory diverges beyond 'cutoff'.

    Args:
        params (dict): Physical parameters dictionary.
        init_xyz (tuple, optional): Initial conditions (x, y, z). Defaults to (0.1, 0.0, 0.0).
        dt (float, optional): Integration time step. Defaults to 0.01.
        t_skip (float, optional): Time to simulate and discard before recording. Defaults to 100.
        t_end (float, optional): Total duration of the recorded simulation. Defaults to 500.
        cutoff (float, optional): Divergence Threshold. System is considered invalid if abs(x) > cutoff. Defaults to 150.

    Returns:
        tuple: Numpy arrays representing the trajectory.
            - x_arr (np.array): X-coordinates of the trajectory.
            - y_arr (np.array): Y-coordinates of the trajectory.
            - z_arr (np.array): Z-coordinates of the trajectory.
    
            
    """
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


def cost_function(param_array, target_featuers, param_names, dt, t_skip, t_end, penalty=1e6):
    """Evaluates the fitness of a parameter set against target observational data.

    Core objective function for the Differential Evolution optimizer. 
    Penalizes trajectories that:
        - are divergent,
        - fail to oscillate (cross the mean),
        - have unphysically small amplitudes,
        - exhibit chaotic/quasi-periodic peak variations.

    Calculates a scalarized Fourier distance between the simulated wave and the target star for valid limit cycles.

    Args:
        param_array (list or array-like): Parameter values being tested by the optimizer in a given iteration.
        target_featuers (dict): The empirical target values to fit.
        param_names (list of str): The string keys corresponding to 'param_array'.
        dt (float): Integration time step.
        t_skip (float): Skips calculations until t_skip.
        t_end (float): Ends calculations at t_end.
        penalty (float, optional): Penalty returned for invalid trajectories. Defaults to 1e6.
    Returns:
        float: Parameter cost/error. Lower score suggests better fit. Returns penalty for invalid trajectories.
    """

    params = {name: val for name, val in zip(param_names, param_array)}
    
    x_array, y_array, z_array = get_trajectory(params, dt=dt, t_skip=t_skip, t_end=t_end)
    if x_array is None: return penalty

    z_mean = np.mean(z_array)
    has_crossed = (z_array[:-1] <= z_mean) & (z_array[1:] > z_mean)
    where_crossed = np.where(has_crossed)[0]
    if len(where_crossed) < 2: return penalty 

    crossed_x = x_array[where_crossed]
    x_amplitude = np.max(x_array) - np.min(x_array)
    if x_amplitude < 0.5: 
        return penalty
    periodicity_error = np.std(crossed_x) / (x_amplitude + 1e-9) 
    
    features = com.get_fourier_features(x_array, dt)
    if features is None: return penalty
        
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
    """Generates an initial population cluster around a starting point.

    Creates a Gaussian cloud of initial points centered on a provided guess instead of 
    starting with uniform distribution across the parameter space. In assumption this 
    should significantly accelerate convergence when tuning a model.

    Args:
        base_params (list or array-like): The central parameter values to build around.
        bounds (list of tuples): The (min, max) absolute limits for each parameter.
        popsize (int, optional): Total number of individuals in the population. Defaults to 100.
        spread_fraction (float, optional): Standard deviation of the Gaussian noise applied to the base parameters, 
                                           expressed as a fraction of the total bound range. Defaults to 0.05 (5%).

    Returns:
        np.ndarray: 2D array (popsize, num_params) containing the initialized population, clipped to keep values within the specified bounds.
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


def main():
    """Executes the inverse-problem optimization pipeline using Differential Evolution.

    1. Reads configuration settings and a starting parameter guess from 'config.toml'. 
    2. Initializes a seeded population.
    3. Runs a parallelized Differential Evolution optimizer to find parameters that best reproduce the target star's empirical Fourier features. 
    4. Evaluates and saves the best fit to the configuration file.
    """
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
                        "R21": 0.244
                        ,"phi21": 4.457       
                        ,"R31": 0.101
                        ,"phi31": 2.222
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
    print(f"R31:   Sim = {final_features['R31']}      | Target={OGLE_TARGETS['R31']}")
    print(f"phi31: Sim = {final_features['phi31']}    | Target={OGLE_TARGETS['phi31']}")

    with open("config.toml", "a", encoding="utf-8") as conf:
        conf.write(f"\n[[SAVED_PARAMS]] # State: OPTIMIZED_FIT_ERR_{result.fun}\n") 
        for p_key, p_val in best_params.items():
            conf.write(f"{p_key} = {p_val}\n")
        print('\n-> BEST FIT SAVED TO: config.toml')

if __name__ == "__main__": main()
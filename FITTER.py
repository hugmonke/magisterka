import numpy as np
from scipy.optimize import differential_evolution
import COMMON as com 
import tomllib
from numba import njit

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
    return com.runge_kutta_scipy(params=params
                                 ,  init_xyz=init_xyz
                                 ,  dt_eval=dt
                                 ,  t_skip=t_skip
                                 ,  t_record=t_end
                                 ,  cutoff=cutoff
                                 )


def cost_function(param_array, target_features, param_names, dt, t_skip, t_end, cutoff, penalty=5e3):
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
        target_features (dict): The empirical target values to fit.
        param_names (list of str): The string keys corresponding to 'param_array'.
        dt (float): Integration time step.
        t_skip (float): Skips calculations until t_skip.
        t_end (float): Ends calculations at t_end.
        penalty (float, optional): Penalty returned for invalid trajectories. Defaults to 1e6.
    Returns:
        float: Parameter cost/error. Lower score suggests better fit. Returns penalty for invalid trajectories.
    """

    params = {name: val for name, val in zip(param_names, param_array)}
    
    is_valid, x_array, y_array, z_array = com.get_trajectory_numba(init_xyz=(0.1, 0.0, 0.0)
                                                          , alpha=params['alpha']
                                                          , mu=params['mu']
                                                          , gamma=params['gamma']
                                                          , p=params['p']
                                                          , s=params['s']
                                                          , dt=dt
                                                          , t_skip=t_skip
                                                          , t_end=t_end
                                                          , cutoff=cutoff
                                                          )
    
    if not is_valid: return penalty

    half_idx = len(x_array) // 2
    first_half = x_array[:half_idx]
    second_half = x_array[half_idx:]
    
    mean_drift = np.abs(np.mean(first_half) - np.mean(second_half))
    amp_first = np.max(first_half) - np.min(first_half)
    amp_second = np.max(second_half) - np.min(second_half)
    amp_drift = np.abs(amp_first - amp_second)
    
    if mean_drift > 0.07 or amp_drift > 0.07:
        return penalty
    
    z_mean = np.mean(z_array)
    has_crossed = (z_array[:-1] <= z_mean) & (z_array[1:] > z_mean)
    where_crossed = np.where(has_crossed)[0]
    if len(where_crossed) < 2: return penalty 
    

    # FIX 1: Exact Interpolation for physical periodicity error
    crossed_x = []
    for idx in where_crossed:
        z1, z2 = z_array[idx], z_array[idx+1]
        x1, x2 = x_array[idx], x_array[idx+1]
        t_cross = (z_mean - z1) / (z2 - z1 + 1e-12)
        x_cross = x1 + t_cross * (x2 - x1)
        crossed_x.append(x_cross)
    
    crossed_x = np.array(crossed_x)
    x_amplitude = np.max(x_array) - np.min(x_array)
    
    # FIX 2: Soft Penalty Slope for dying amplitude
    if x_amplitude < 0.5: 
        return 1000.0 / (x_amplitude + 1e-9)
    
    periodicity_error = np.std(crossed_x) / (x_amplitude + 1e-9) 
    
    features = com.get_fourier_features(x_array, dt)
    if features is None: return penalty
        
    error_R21 = (features["R21"] - target_features["R21"])**2
    diff_phi = abs(features["phi21"] - target_features["phi21"])
    diff_phi = min(diff_phi, 2 * np.pi - diff_phi)
    error_phi21 = diff_phi**2

    error_R31 = 0
    error_phi31 = 0
    if "R31" in target_features and "phi31" in target_features:
        error_R31 = (features["R31"] - target_features["R31"])**2
        diff_phi31 = abs(features["phi31"] - target_features["phi31"])
        error_phi31 = (min(diff_phi31, 2 * np.pi - diff_phi31))**2
    

    # R values are tiny (0.1 - 0.5), Phases are large (0 - 6.28)
    # We multiply the R errors by 10 to make the algorithm care about them equally.
    total_cost = periodicity_error + 100*error_R21 + error_phi21 + 100*error_R31 + error_phi31
    
    return total_cost


def create_seeded_population(base_params, bounds, popsize=100, spread_fraction=0.05):
    """DEPRECATED! Generates an initial population cluster around a starting point.

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

def find_test(dt, t_skip, t_end, cutoff, param_arr, spread, bounds, maxiter, popsize, fit_threshold):
    try:
        with open("test_stars.toml", "rb") as test_config:
            test_stars_data = tomllib.load(test_config)
            raw_test_stars = test_stars_data.get("TEST_STAR", [])
    except FileNotFoundError:
        raw_test_stars = False
    if raw_test_stars:
        print(f"Starting test benchmark on {len(raw_test_stars)} test stars.")
        found_count = 0
        
        for i, test_params in enumerate(raw_test_stars):
            test_star_label = f'Test_Star_{i+1}'
            print(f">>> Fitting {test_star_label}")
            print(f"Parameteres: {test_params}")
            # ==================================================================================
            # We dont need to do it once again, just get fourier features from MAP_MAKER
            is_valid, x_target, _, _ = com.get_trajectory_numba(init_xyz=(0.1, 0.0, 0.0)
                                                                , alpha=test_params['alpha']
                                                                , mu=test_params['mu']
                                                                , gamma=test_params['gamma']
                                                                , p=test_params['p']
                                                                , s=test_params['s']
                                                                , dt=dt
                                                                , t_skip=t_skip
                                                                , t_end=t_end
                                                                , cutoff=cutoff
                                                                )
            if not is_valid:
                print(f"SKIPPED: Parameters for {test_star_label} diverged.\n")
                continue

            target_features = com.get_fourier_features(x_target, dt=dt)
            if not target_features:
                print(f"SKIPPED: Could not extract Fourier features for {test_star_label}.\n")
                continue
            # ==================================================================================
            print(f"Extracted features for {test_star_label}: \
                  R21 = {target_features['R21']} \
                , phi21 = {target_features['phi21']} \
                , R31 = {target_features.get('R31', 0.0)} \
                , phi31 = {target_features.get('phi31', 0.0)}")
            
            neighbours_list = test_stars_data.get(f"TEST_STAR_{i}_NEIGHBOUR", [{}])
            if neighbours_list and neighbours_list[0]:
                guess_params = [neighbours_list[0][k] for k in param_arr]
                benchmark_bounds = []
                for guess, (bound_min, bound_max) in zip(guess_params, bounds):
                    range_span = bound_max - bound_min
                    benchmark_bounds.append((max(bound_min, guess - range_span * spread), min(bound_max, guess + range_span * spread)))
            else:
                benchmark_bounds = bounds
            print(f"Performing differential evolution on {test_star_label}\n")
            result = differential_evolution(func=cost_function
                                            , bounds=benchmark_bounds
                                            , args=(target_features, param_arr, dt, t_skip, t_end, cutoff)
                                            , strategy='randtobest1exp'
                                            , maxiter=maxiter
                                            , popsize=popsize
                                            , mutation=(0.7, 1.5)
                                            , recombination=0.9
                                            , init='latinhypercube'
                                            , tol=0.001
                                            , polish=False
                                            , disp=False  # Keep console clean during benchmark
                                            , workers=-1
                                            , updating='deferred'
                                            )
            
            if result.fun <= fit_threshold:
                print(f"GOOD TRAJECTORY FOUND! Final Error: {result.fun}\n")
                found_count += 1
                best_params = {name: val for name, val in zip(param_arr, result.x)}

                with open("test_stars.toml", "a", encoding="utf-8") as tconf:
                    tconf.write(f"[[TEST_STAR_{i}_RECOVERED]] # ERROR: {result.fun}\n")
                    for p_key, p_val in best_params.items():
                        tconf.write(f"{p_key} = {p_val}\n")
            else:
                print(f"NO GOOD TRAJECTORY FOUND. Final Error: {result.fun}\n")
                with open("test_stars.toml", "a", encoding="utf-8") as tconf:
                    tconf.write(f"[[TEST_STAR_{i}_RECOVERED]] # BAD FIT | ERROR: {result.fun}\n")
                    for p_key, p_val in best_params.items():
                        tconf.write(f"{p_key} = {p_val}\n")

        print(f"==========================================")
        print(f"BENCHMARK COMPLETE: Found {found_count} out of {len(raw_test_stars)} stars.")
        print(f"==========================================")
    else:
        "No test stars given in test_stars.toml. Skipping."
        return

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
    BOUNDS          = [
                        (-6, 9),       # alpha
                        (-6, 9),       # mu
                        (0, 7),        # gamma
                        (0, 7),        # p
                        (0, 7)         #
                       ]
    DT              = config.get("DT", 0.01)
    T_SKIP          = config.get("T_SKIP", 500)
    T_END           = config.get("T_END", 1000)
    SPREAD          = config.get("SPREAD", 0.01)
    POPSIZE         = config.get("POPSIZE", 100)
    CUTOFF          = config.get("CUTOFF", 1e6)
    MAXITER         = config.get("MAXITER", 1000)
    REAL_PARAMS     = config.get("SAVED_STAR_PARAMS", [{}])[0]
    FIT_THRESHOLD   = config.get("FIT_TRESHOLD", 0.01)
    FIND_TEST = True
    FIND_REAL = False
    PARAM_ARR       = ['alpha', 'mu', 'gamma', 'p', 's']
    
    # OGLE_TARGETS    = {"R21": 0.367, "phi21": 5.342, "R31": 0.184, "phi31": 3.578}
    # OGLE_TARGETS    = {'R21': 0.447, 'phi21': 4.738, 'R31': 0.206, 'phi31': 3.168} #00002
    OGLE_TARGETS    = {'R21': 0.179, 'phi21': 0.596, 'R31': 0.055, 'phi31': 0.818} #36125

    print("Starting Differential Evolution Optimizer\n")
    if FIND_TEST:
        print("=== TEST DATASET: Tanaka-Takeuti Inverse Problem ===") 
        find_test(dt=DT
                  , t_skip=T_SKIP
                  , t_end=T_END
                  , cutoff=CUTOFF
                  , param_arr=PARAM_ARR
                  , spread=SPREAD
                  , bounds=BOUNDS
                  , maxiter=MAXITER
                  , popsize=POPSIZE
                  , fit_threshold=FIT_THRESHOLD)
    if FIND_REAL:
        print("=== REAL DATASET: Tanaka-Takeuti Inverse Problem ===")
        print(f"Targeting OGLE RRab: R21={OGLE_TARGETS['R21']}, phi21={OGLE_TARGETS['phi21']}, R31={OGLE_TARGETS.get(('R31'), None)}, phi31={OGLE_TARGETS.get(('phi31'), None)}")
        print(f"Looking around parameters: {REAL_PARAMS}")
        MAP_MAKER_GUESS = [REAL_PARAMS[k] for k in PARAM_ARR]
        tight_bounds = []
        for guess, (bound_min, bound_max) in zip(MAP_MAKER_GUESS, BOUNDS):
            range_span = bound_max - bound_min
            tight_bound_min = max(bound_min, guess - range_span * SPREAD)
            tight_bound_max = min(bound_max, guess + range_span * SPREAD)
            tight_bounds.append((tight_bound_min, tight_bound_max))

        result = differential_evolution(func=cost_function
                                        , bounds=tight_bounds
                                        , args=(OGLE_TARGETS, PARAM_ARR, DT, T_SKIP, T_END, CUTOFF)
                                        , strategy='randtobest1exp'
                                        , maxiter=MAXITER
                                        , popsize=POPSIZE
                                        , mutation=(0.7, 1.5)
                                        , recombination=0.9
                                        , init='latinhypercube'
                                        , tol=0.001
                                        , polish=False
                                        , disp=True
                                        , workers=-1
                                        , updating='deferred'
                                        )
        
        print(f"=== OPTIMIZATION FINISHED (Final Error: {result.fun:.5f}) ===")
        if result.fun < 0.01:
            print("> RESULT: GOOD FIT")
        elif result.fun < 0.2:
            print("> RESULT: MID FIT")
        else:
            print("> RESULT: BAD FIT")
            
        best_params = {name: val for name, val in zip(PARAM_ARR, result.x)}
        final_x, final_y, final_z = get_trajectory(best_params, dt=DT, t_skip=T_SKIP, t_end=T_END, cutoff=CUTOFF)
        if final_x is None:
            print("WARNING: The optimizer's best result is DIVERGENT. Run the optimizer again.") 
            return
        final_features = com.get_fourier_features(final_x, dt=DT)
        
        print("=== Best Parameters Found ===\n")
        for k, v in best_params.items():
            print(f"{k:>6}: {v:.4f}")
        print("=== Simulation vs Target ===\n")
        print(f"R21:   Sim = {final_features['R21']}      | Target = {OGLE_TARGETS['R21']}")
        print(f"phi21: Sim = {final_features['phi21']}    | Target = {OGLE_TARGETS['phi21']}")
        print(f"R31:   Sim = {final_features['R31']}      | Target = {OGLE_TARGETS['R31']}")
        print(f"phi31: Sim = {final_features['phi31']}    | Target = {OGLE_TARGETS['phi31']}")

        with open("config.toml", "a", encoding="utf-8") as conf:
            conf.write(f"\n[[SAVED_PARAMS]] # State: OPTIMIZED_FIT_ERR_{result.fun}\n") 
            for p_key, p_val in best_params.items():
                conf.write(f"{p_key} = {p_val}\n")
            print('\n-> BEST FIT SAVED TO: config.toml')

if __name__ == "__main__": main()
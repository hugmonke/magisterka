import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import COMMON as com 
import tomllib

def generate_dataset(filename, sim_num = 1000, print_every = 50, init_xyz = (0.1, 0.0, 0.0), params = None, dt = 0.01, t_skip = 50, t_end = 150, size = 1000, cutoff = 1e6, batch_size = 1000, tolerance = 0.1):
    """Generates a dataset by running batched simulations of the dynamical system.

    Sweeps the parameter space by running multiple simulations in batches. 
    Calculates the Largest Lyapunov Exponent (LLE) for each valid trajectory.
    Extracts the Poincaré section, computes the Shannon entropy, and evaluates the Fourier features. 
    Classifies the dynamical state (CHAOTIC, PERIODIC, QUASI-PERIODIC, STABLE, DIVERGENT) and logs all results to a local text file.

    Args:
        filename (str): Log file path.
        sim_num (int, optional): Total number of simulations. Defaults to 1000.
        print_every (int, optional): Frequency of console progress updates. Defaults to 50.
        init_xyz (tuple, optional): Initial conditions (x, y, z) for the system. Defaults to (0.1, 0.0, 0.0).
        dt (float, optional): Integration time step. Defaults to 0.01.
        t_skip (float, optional): Simulation time to discard. Defaults to 50.
        t_end (float, optional): Total simulation time per run. Defaults to 150.
        size (int, optional): Size of the generated parameter array. Defaults to 1000.
        cutoff (float, optional): Physical divergence threshold. Defaults to 1e6.
        batch_size (int, optional): Number of simulations to process in one batch. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame containing parameters, dynamical states, entropy, LLE, and Fourier features for all processed simulations.
    """
    print(f"Running {sim_num} simulations to map parameter space")
    overflow_count = 0
    results = []
    
    with open(filename, "a", encoding="utf-8") as file:
        total_sim = 0
        print("Calculating trajectories and entropy, saving results to sim_results_space.txt")
        if params is None or params == {}:
            master_params = com.get_parameteres(size=sim_num)
        else:
            master_params = {param: np.array(val).flatten() for param, val in params.items()}

        while total_sim < sim_num:
            cur_size = min(batch_size, sim_num - total_sim)
            batch_params = {k: v[total_sim : total_sim + cur_size] for k, v in master_params.items()}

            cur_size = min(batch_size, sim_num - total_sim)
            params = com.get_parameteres(size=cur_size)
            lle_all, valid_mask, x_all, y_all, z_all = com.solve_and_get_lle(init_xyz = init_xyz
                                                                            , params = batch_params
                                                                            , dt = dt
                                                                            , t_skip = t_skip
                                                                            , t_end = t_end
                                                                            , size = cur_size
                                                                            , cutoff= cutoff
                                                                            )
        
            for i in range(cur_size):
                no_trajectory = i + total_sim
                run_data = {param: val[i] for param, val in params.items()}
                param_string = ", ".join([f"'{param}': {val}" for param, val in run_data.items()])
                R21, PHI21, R31, PHI31 = np.nan, np.nan, np.nan, np.nan
                if not valid_mask[i]:
                    overflow_count += 1
                    state = "DIVERGENT"
                    entropy = -1.0
                    lle = np.nan
                else:
                    x, y, z = x_all[:, i], y_all[:, i], z_all[:, i]
                    lle = lle_all[i]
                    
                    alpha, mu = run_data['alpha'], run_data['mu']
                    gamma, p, s = run_data['gamma'], run_data['p'], run_data['s']
                    
                    mean_pos = (np.mean(x), np.mean(y), np.mean(z))
                    v_vec = com.get_derivatives_numba(x[0], y[0], z[0], alpha, mu, gamma, p, s)
                    plane = com.generate_plane(point=mean_pos, normal=v_vec)
                    poinc_x, poinc_y, _ = com.poincare_map(x=x, y=y, z=z, plane=plane)

                    entropy = com.shannon_entropy(poinc_x, poinc_y)
                    state = com.classify(entropy, lle)
                    
                    state, features = com.validate_state_and_features(x_array=x, dt=dt, state=state, tolerance=tolerance)
                    R21, PHI21 = features["R21"], features["phi21"]
                    R31, PHI31 = features["R31"], features["phi31"]

                    run_data.update({"Entropy": entropy, "LLE": lle, "State": state, "R21": R21, "phi21": PHI21, "R31": R31, "phi31": PHI31})
                results.append(run_data)
                log_line = f"Classified State: {state:<14} | Entropy: {entropy:>7.4f} | LLE: {lle:>7.4f} | Params: {{{param_string}}} | R21: {R21:>6.3f} | phi21: {PHI21:>6.3f} | R31: {R31:>6.3f} | phi31: {PHI31:>6.3f} | T_SKIP: {t_skip} | T_END: {t_end}\n"
                file.write(log_line)
                if (no_trajectory+1) % print_every == 0:
                    print(f"Run no. {no_trajectory+1} ")
            
            total_sim += cur_size

    print(f"Total successful runs: {len(results)}")
    print(f"Total exploded (error) runs: {overflow_count}")
    print("-> Results saved to sim_results_space.txt")
    return pd.DataFrame(results)


def train_rf_classifier(df_params, df_states, features, n_estimators, test_size, random_state):
    """Trains a Random Forest Classifier to predict dynamical states from the chosen model parameters.

    Splits the provided parameter dataset into training and testing subsets, fits a 
    Random Forest Classifier to predict the resulting dynamical behavior and prints 
    classification accuracy and relative importance of each parameter.

    Args:
        df_params (pd.DataFrame): Input parameters.
        df_states (pd.Series): Target dynamical states.
        features (list of str): Names of parameters being evaluated.
        n_estimators (int): Number of decision trees in the random forest.
        test_size (float): Percentage of dataset to include in test sample.
        random_state (int): Seed value for reproducibility.

    Returns:
        None: Prints classification accuracy and feature importances to the console.
    """
    print("=== Feature Importances ===")
    X_train, X_test, Y_train, Y_test = train_test_split(df_params, df_states, test_size=test_size, random_state=random_state)

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(X_train, Y_train)

    accuracy = rf_classifier.score(X_test, Y_test)
    print(f"Random Forest Classification Accuracy: {accuracy*100:.2f}%\n")

    importances = rf_classifier.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    for idx in sorted_indices:
        print(f"{features[idx]:>5}: {importances[idx]*100:.2f}%")



def main():
    """Executes the parameter space mapping and machine learning pipeline.

    1. Reads configuration settings from 'config.toml'.
    2. Generates a dataset of classifications of simulated stellar pulsations and their parameteres.
    3. (Optionally) Plots 2D UMAP visulisation of the parameter space.
    4. (Optionally) Trains Random Forest classifier for parameter importance analysis.

    """
    try:
        with open("config.toml", "rb") as conf:
            config = tomllib.load(conf)
    except FileNotFoundError as e:
        print(f"{e}: config.toml not found.")
        exit()

    # COMMON
    PARAMS          = config.get("PARAMS", None)
    INIT_XYZ        = config.get("INIT_XYZ", [0.1, 0.0, 0.0])
    DT              = config.get("DT", 0.01)
    T_SKIP          = config.get("T_SKIP", 100)
    T_END           = config.get("T_END", 1000)
    CUTOFF          = config.get("CUTOFF", 1e6)
    TOLERANCE       = config.get("TOLERANCE", 0.05)
    SIM_NUM         = config.get("SIM_NUM", 1_000)
    PRINT_EVERY     = config.get("PRINT_EVERY", 50) 
    BATCH_SIZE      = config.get("BATCH_SIZE", 1000) 
    FILENAME        = config.get("FILENAME", "sim_results_space.txt") 
    
    # UMAP PARAMETERS:
    N_NEIGHBORS     = config.get("N_NEIGHBORS", 15) # n_neighbors controls how UMAP balances local vs global structure.
    MIN_DIST        = config.get("MIN_DIST", 0.1)   # min_dist controls how tightly points are packed together.
    RANDOM_STATE    = config.get("RANDOM_STATE", 1)

    # RANDOM FOREST PARAMETERS:
    N_ESTIMATORS    = config.get("N_ESTIMATORS", 100)
    TEST_SIZE       = config.get("TEST_SIZE", 0.2)

    # PLOT PARAMETERS
    PLOT_PARAM_SPACE = False
    TRAIN_CLASSIFIER = False


    dataset = generate_dataset(filename=FILENAME
                        , sim_num = SIM_NUM
                        , print_every = PRINT_EVERY
                        , init_xyz = INIT_XYZ
                        , params = PARAMS
                        , dt = DT
                        , t_skip = T_SKIP
                        , t_end = T_END
                        , size = SIM_NUM
                        , cutoff = CUTOFF
                        , batch_size = BATCH_SIZE
                        , tolerance = TOLERANCE
                        )
    
    if dataset.empty or len(dataset['State'].unique()) < 2:
        print("Not enough varied data. Increase number of simulations (sim_num).")
        exit()

    priority_dict = {"DIVERGENT": 0
                      , "STABLE": 1
                      , "QUASI_PERIODIC": 2
                      , "PERIODIC": 3
                      , "CHAOTIC": 4
                      }
    
    dataset['priority'] = dataset['State'].map(priority_dict).fillna(-1)
    dataset = dataset.sort_values('priority').drop(columns=['priority']).reset_index(drop=True)
    param_arr = ['alpha', 'mu', 'gamma', 'p', 's']
    df_params, df_states = dataset[param_arr], dataset['State']


    if PLOT_PARAM_SPACE:
        com.plot_parameter_space(df_params=df_params
                             , df_states=df_states
                             , n_neighbors=N_NEIGHBORS
                             , min_dist=MIN_DIST
                             , random_state=RANDOM_STATE
                             )
    
    if TRAIN_CLASSIFIER:
        train_rf_classifier(df_params=df_params
                            , df_states=df_states
                            , features=param_arr
                            , n_estimators=N_ESTIMATORS
                            , test_size=TEST_SIZE
                            , random_state=RANDOM_STATE
                            )



if __name__ == "__main__": main()
    
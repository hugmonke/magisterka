import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import COMMON as com 
import tomllib
from scipy.fft import rfft, rfftfreq

def get_fourier_features(x_array, dt):
    """Performs FFT on the simulated trajectory to extract R21 and phi21."""

    N = len(x_array)
    x_centered = x_array - np.mean(x_array) # center wave
    fft_vals = rfft(x_centered)
    amps = np.abs(fft_vals)/N
    phases = np.angle(fft_vals)
    freqs = rfftfreq(N, dt)
    
    # FUNDAMENTAL FREQUENCY
    idx_1 = np.argmax(amps[1:]) + 1
    f_1 = freqs[idx_1]
    A_1 = amps[idx_1]
    phi_1 = phases[idx_1]
    
    if A_1 < 1e-10: return None # 1e-10 allows small wave survival
        
    # HARMONIC
    idx_2 = np.argmin(np.abs(freqs - 2 * f_1))
    A_2 = amps[idx_2]
    phi_2 = phases[idx_2]
    
    R21 = A_2 / A_1
    phi21 = (phi_2 - 2 * phi_1) % (2 * np.pi)
    
    return {"R21": R21, "phi21": phi21}


# ------------------------------------------------------------
# DATA GENERATION
# ------------------------------------------------------------
def get_dataset(filename, sim_num = 1000, print_every = 50, init_xyz = (0.1, 0.0, 0.0), params = None, dt = 0.01, t_skip = 50, t_end = 150, size = 1000, cutoff= 1e6, batch_size = 1000):
    print(f"Running {sim_num} simulations to map parameter space")
    overflow_count = 0
    results = []
    
    with open(filename, "a", encoding="utf-8") as file:
        total_sim = 0
        print("Post-processing trajectories and calculating entropy, saving to sim_results_space.txt")
        while total_sim < sim_num:

            cur_size = min(batch_size, sim_num - total_sim)
            params = com.get_parameteres(params=None, size=cur_size)
            lle_all, valid_mask, x_all, y_all, z_all = com.solve_and_get_lle(init_xyz = init_xyz
                                                                            , params = params
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
                # Skip if the simulation exploded
                if not valid_mask[i]:
                    overflow_count += 1
                    state = "DIVERGENT"
                    entropy = np.nan
                    lle = np.nan
                else:
                    x, y, z = x_all[:, i], y_all[:, i], z_all[:, i]
                    lle = lle_all[i]
                    
                    z_mean = np.mean(z)
                    crossed = (z[:-1] - z_mean) * (z[1:] - z_mean) < 0
                    crossed_ind = np.where(crossed)[0] + 1
                    poinc_x = x[crossed_ind]
                    poinc_y = y[crossed_ind]

                    entropy = com.shannon_entropy(poinc_x, poinc_y)
                    state = com.classify(entropy, lle)
                    
                    R21, PHI21 = np.nan, np.nan # Default
                    
                    if state == "PERIODIC":
                        # Double check if PERIODIC, not DIVERGENT (The wave doesnt run away)
                        drift_threshold = 0.05 # Chosen arbitrally, works best, allows some numerical errors
                        half_idx = len(x)//2 
                        drift_ratio = abs(np.mean(x[:half_idx]) - np.mean(x[half_idx:])) / (np.max(x) - np.min(x) + 1e-9)
                        if drift_ratio < drift_threshold: 
                            features = get_fourier_features(x, dt=dt) 
                            if features: R21, PHI21 = features["R21"], features["phi21"]
                            else: state = "DIVERGENT"
                    
                run_data.update({"Entropy": entropy, "LLE": lle, "State": state, "R21": R21, "phi21": PHI21})
                results.append(run_data)
                log_line = f"Classified State: {state:<14} | Entropy: {entropy:>7.4f} | LLE: {lle:>7.4f} | Params: {{{param_string}}} | R21: {R21:>6.3f} | phi21: {PHI21:>6.3f} | T_SKIP: {t_skip} | T_END: {t_end}\n"
                file.write(log_line)
                if (no_trajectory+1) % print_every == 0:
                    print(f"Run no. {no_trajectory+1} ")
            
            total_sim += cur_size

    print(f"Total successful runs: {len(results)}")
    print(f"Total exploded (error) runs: {overflow_count}")
    print("-> Results saved to sim_results_space.txt")
    return pd.DataFrame(results)


def train_rf_classifier(df_params, df_states, features, n_estimators, test_size, random_state):
    """Trains a Random Forest and prints feature importances."""
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
    PLOT_PARAM_SPACE = True
    PLOT_FOURIER_SPACE = True
    TRAIN_CLASSIFIER = False

    # FOURIER SPACE PARAMETERS
    TARGET_R21 = 0.545
    TARGET_PHI21 = 4.395 
    MODEL_LABEL = 'Tanaka-Takeuti Model'
    STAR_LABEL = 'OGLE-LMC-RRLYR-00001'

    dataset = get_dataset(filename=FILENAME
                        , sim_num = SIM_NUM
                        , print_every = PRINT_EVERY
                        , init_xyz = INIT_XYZ
                        , params = PARAMS
                        , dt = DT
                        , t_skip = T_SKIP
                        , t_end = T_END
                        , size = SIM_NUM
                        , cutoff = CUTOFF
                        , batch_size= BATCH_SIZE
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

    if PLOT_FOURIER_SPACE:
        com.plot_fourier_space(dataset
                           , target_R21=TARGET_R21
                           , target_phi21=TARGET_PHI21
                           , model_label=MODEL_LABEL
                           , star_label=STAR_LABEL
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
    
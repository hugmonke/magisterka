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

# ------------------------------------------------------------
# DATA GENERATION
# ------------------------------------------------------------
def get_dataset(sim_num = 1000, print_every = 50, init_xyz = (0.1, 0.0, 0.0), params = None, dt = 0.01, t_skip = 50, t_end = 150, size = 1000, cutoff= 1e6, batch_size = 1000):
    print(f"Running {sim_num} simulations to map parameter space")
    overflow_count = 0
    results = []
    with open("sim_results_space.txt", "a", encoding="utf-8") as file:
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
                        
                    
                run_data.update({"Entropy": entropy, "LLE": lle, "State": state})
                results.append(run_data)
                log_line = f"Classified State: {state:<14} | Entropy: {entropy:>7.4f} | LLE: {lle:>7.4f} | Params: {{{param_string}}} | T_SKIP: {t_skip} | T_END: {t_end}\n"
                file.write(log_line)
                if (no_trajectory+1) % print_every == 0:
                    print(f"Run no. {no_trajectory+1} ")
            
            total_sim += cur_size

    print(f"Total successful runs: {len(results)}")
    print(f"Total exploded (error) runs: {overflow_count}")
    print("-> Results saved to sim_results_space.txt")
    return pd.DataFrame(results)


def plot_parameter_space(df_params, df_states, n_neighbors, min_dist, random_state):
    """Scaling, UMAP dimensionality reduction and UMAP plotting."""
    print("=== Running UMAP projection ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_params)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(X_scaled)
    
    # PARAMETER SPACE PLOT (UMAP Projection)
    fig, ax = plt.subplots(figsize=(12, 8))
    color_map = {"CHAOTIC": "red"
                 , "STABLE": "green"
                 , "PERIODIC": "blue"
                 , "QUASI_PERIODIC": "purple"
                 , "DIVERGENT": "orange"
                 }
    colors = df_states.map(color_map).fillna('black').tolist()
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, marker='s', s=5, picker=True, pickradius=5)

    for state_type, color in color_map.items():
        if state_type in df_states.values:
            ax.scatter([], [], c=color, label=state_type, s=20)

    def on_pick(event):
        # If points overlap, pick the first element from this list
        idx = event.ind[0] 
        # Get parameters and classified state
        param = df_params.iloc[idx]
        state = df_states.iloc[idx]
        param_dict = param.to_dict()
        print('=========================\n\n')
        print(f"[{state} POINT CLICKED]\n")
        print(param_dict)
        with open("config.toml", "a", encoding="utf-8") as conf:
            conf.write(f"\n[[SAVED_PARAMS]] # State: {state}\n") 
            for param, val in param_dict.items():
                conf.write(f"{param} = {val}\n")
        print('POINT SAVED TO: config.toml')
        print('=========================\n\n')

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    plt.title("Parameter Space Mapping")
    plt.xlabel("DIM 1")
    plt.ylabel("DIM 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_rf_classifier(df_params, df_states, features, n_estimators, test_size, random_state):
    """Trains a Random Forest and prints feature importances."""

    X_train, X_test, Y_train, Y_test = train_test_split(df_params, df_states, test_size=test_size, random_state=random_state)

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(X_train, Y_train)

    accuracy = rf_classifier.score(X_test, Y_test)
    print(f"Random Forest Classification Accuracy: {accuracy*100:.2f}%\n")

    print("=== Feature Importances ===")
    importances = rf_classifier.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    for idx in sorted_indices:
        print(f"{features[idx]:>5}: {importances[idx]*100:.2f}%")



def main():
    with open("config.toml", "rb") as conf:
        config = tomllib.load(conf)

    PARAMS          = config.get("PARAMS", None)
    INIT_XYZ        = config.get("INIT_XYZ", [0.1, 0.0, 0.0])
    DT              = config.get("DT", 0.01)
    T_SKIP          = config.get("T_SKIP", 100)
    T_END           = config.get("T_END", 1000)
    CUTOFF          = config.get("CUTOFF", 1e6)
    SIM_NUM         = config.get("SIM_NUM", 1_000)
    PRINT_EVERY     = config.get("PRINT_EVERY", 50) 
    BATCH_SIZE      = config.get("BATCH_SIZE", 1000) 
    
    # UMAP PARAMETERS:
    N_NEIGHBORS     = config.get("N_NEIGHBORS", 15) # n_neighbors controls how UMAP balances local vs global structure.
    MIN_DIST        = config.get("MIN_DIST", 0.1)   # min_dist controls how tightly points are packed together.
    RANDOM_STATE    = config.get("RANDOM_STATE", 1)

    # RANDOM FOREST PARAMETERS:
    N_ESTIMATORS    = config.get("N_ESTIMATORS", 100)
    TEST_SIZE       = config.get("TEST_SIZE", 0.2)

    dataset = get_dataset(sim_num = SIM_NUM
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

    param_str = ['alpha', 'mu', 'gamma', 'p', 's']
    df_params, df_states = dataset[param_str], dataset['State']

    plot_parameter_space(df_params=df_params, df_states=df_states, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, random_state=RANDOM_STATE)
    train_rf_classifier(df_params=df_params, df_states=df_states, features=param_str, n_estimators=N_ESTIMATORS, test_size=TEST_SIZE, random_state=RANDOM_STATE)



if __name__ == "__main__": main()
    
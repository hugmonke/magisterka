import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import tomllib
import ast
import re
import os

def parse_log_file(filepath="sim_results_space.txt"):
    """Reads the simulation log and extracts parameters and states."""
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find {filepath}")
        exit()

    print(f"Reading data from {filepath}")
    data = []
    
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): 
                continue
                
            state_match = re.search(r"Classified State:\s*([A-Z_]+)", line)
            params_match = re.search(r"Params:\s*(\{.*?\})", line)
            
            if state_match and params_match:
                state = state_match.group(1)
                params_str = params_match.group(1)
                try:
                    params_dict = ast.literal_eval(params_str)
                    params_dict['State'] = state
                    data.append(params_dict)
                except Exception as e:
                    print(f"ERROR {e}. Failed to parse line: {line.strip()} ")
                    
    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} simulation runs.")
    return df

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
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, edgecolors='w', s=20, picker=True, pickradius=5)

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


def main():
    try:
        with open("config.toml", "rb") as conf:
            config = tomllib.load(conf)
    except FileNotFoundError as e:
        print(f"{e}: config.toml not found.")
        exit()

    N_NEIGHBORS  = config.get("N_NEIGHBORS", 15) 
    MIN_DIST     = config.get("MIN_DIST", 0.1)  
    RANDOM_STATE = config.get("RANDOM_STATE", 1)
    filepath = "sim_results_space.txt"

    dataset = parse_log_file(filepath)
    if dataset.empty:
        print(f"{filepath} is empty. Exiting.")
        return
    param_str = ['alpha', 'mu', 'gamma', 'p', 's']
    missing_cols = [col for col in param_str if col not in dataset.columns]
    if missing_cols:
        print(f"ERROR: Log file is missing the following columns: {missing_cols}")
        return

    df_params = dataset[param_str]
    df_states = dataset['State']

    plot_parameter_space(df_params=df_params
                         , df_states=df_states
                         , n_neighbors=N_NEIGHBORS
                         , min_dist=MIN_DIST
                         , random_state=RANDOM_STATE
                         )

if __name__ == "__main__":
    main()
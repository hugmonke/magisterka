import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import tomllib
import ast
import re
import os
import COMMON as com


def check_missing_cols(dataset, param_arr):
    missing_cols = [col for col in param_arr if col not in dataset.columns]
    if missing_cols:
        print(f"ERROR: Log file is missing the following columns: {missing_cols}")
        return
    

def print_and_save_neighbours(dataset, param_arr, R21, phi21):
    df_periodic = dataset[(dataset['State'] == 'PERIODIC') & (dataset['R21'].notna())].copy()
    if not df_periodic.empty:
        df_periodic['FOURIER_DIST'] = \
            np.sqrt(
            (df_periodic['R21'] - R21)**2 + 
            ((df_periodic['phi21'] - phi21) / (2*np.pi))**2
            )
        
        TOP_N = 3
        best_matches = df_periodic.sort_values('FOURIER_DIST').head(TOP_N)
        with open("config.toml", "a", encoding="utf-8") as config:
            for i in range(TOP_N):
                match = best_matches.iloc[0]
                config.write(f"\n[[SAVED_PARAMS]] # TOP {i} | DIST: {match['FOURIER_DIST']}\n") 
                for j in param_arr:
                    config.write(f"{j} = {match[j]}\n")

        print(f"=== TOP {TOP_N} NEAREST NEIGHBOURS ===\n")
        print(best_matches[['alpha', 'mu', 'gamma', 'p', 's', 'R21', 'phi21', 'FOURIER_DIST']], "\n")
        print(f"{TOP_N} NEAREST NEIGHBOUR PARAMS SAVED TO CONFIG\n")


def parse_log_file(filepath="sim_results_space.txt"):
    """Reads the simulation log and extracts parameters, states, and Fourier features."""
    print(f"Reading data from {filepath}")
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find {filepath}")
        exit()

    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): 
                continue
                
            state_match = re.search(r"Classified State:\s*([A-Z_]+)", line)
            params_match = re.search(r"Params:\s*(\{.*?\})", line)
        
            r21_match = re.search(r"R21:\s*([^\s\|]+)", line)
            phi21_match = re.search(r"phi21:\s*([^\s\|]+)", line)
            
            if state_match and params_match:
                state = state_match.group(1)
                params_str = params_match.group(1)
                try:
                    params_dict = ast.literal_eval(params_str)
                    params_dict['State'] = state
                    if r21_match and phi21_match:
                        r21_val = r21_match.group(1)
                        phi21_val = phi21_match.group(1)
                        params_dict['R21'] = float(r21_val) if r21_val != "nan" else np.nan
                        params_dict['phi21'] = float(phi21_val) if phi21_val != "nan" else np.nan
                    else:
                        params_dict['R21'] = np.nan
                        params_dict['phi21'] = np.nan
                    data.append(params_dict)
                except Exception as e:
                    print(f"ERROR {e}. Failed to parse line: {line.strip()} ")
                    
    df = pd.DataFrame(data)
    print(f"SUCCESS: LOADED {len(df)} SIMULATION RUNS")
    return df


def main():
    try:
        with open("config.toml", "rb") as config:
            config = tomllib.load(config)
    except FileNotFoundError as e:
        print(f"{e}: config.toml not found.")
        exit()

    # COMMON
    FILENAME = config.get("FILENAME", "sim_results_space.txt") 

    # UMAP PARAMETERS:
    N_NEIGHBORS  = config.get("N_NEIGHBORS", 15) 
    MIN_DIST     = config.get("MIN_DIST", 0.1)  
    RANDOM_STATE = config.get("RANDOM_STATE", 1)

    # PLOT PARAMETERS
    PLOT_PARAM_SPACE = True
    PLOT_FOURIER_SPACE = True

    # FOURIER SPACE PARAMETERS
    # TARGET_R21 = 0.545
    # TARGET_PHI21 = 4.395 
    # MODEL_LABEL = 'Tanaka-Takeuti Model'
    # STAR_LABEL = 'OGLE-LMC-RRLYR-00001'

    TARGET_R21 = 0.101
    TARGET_PHI21 = 0.098
    MODEL_LABEL = 'Tanaka-Takeuti Model'
    STAR_LABEL = 'OGLE-LMC-RRLYR-00029'

    dataset = parse_log_file(FILENAME)
    if dataset.empty:
        print(f"{FILENAME} is empty. Exiting.")
        return
    
    priority_dict = {"DIVERGENT": 0
                    , "STABLE": 1
                    , "QUASI_PERIODIC": 2
                    , "PERIODIC": 3
                    , "CHAOTIC": 4
                    }

    dataset['priority'] = dataset['State'].map(priority_dict).fillna(-1)
    dataset = dataset.sort_values('priority').drop(columns=['priority']).reset_index(drop=True)
    param_arr = ['alpha', 'mu', 'gamma', 'p', 's']
    check_missing_cols(dataset=dataset, param_arr=param_arr)
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

        print_and_save_neighbours(dataset=dataset
                                  , param_arr=param_arr
                                  , R21=TARGET_R21
                                  , phi21=TARGET_PHI21
                                  )


if __name__ == "__main__": main()
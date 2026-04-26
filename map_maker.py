import pandas as pd
import numpy as np
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
    

def print_and_save_neighbours(dataset, param_arr, star_label, R21, phi21, R31=0, phi31=0,TOP_N=1):
    df_periodic = dataset[(dataset['R21'].notna())].copy()
    if not df_periodic.empty:
        diff_phi21 = np.abs(df_periodic['phi21'] - phi21)
        wrap_phi21 = np.minimum(diff_phi21, 2*np.pi - diff_phi21)
        
        diff_phi31 = np.abs(df_periodic['phi31'] - R31) # assuming you meant phi31 here in original
        wrap_phi31 = np.minimum(diff_phi31, 2*np.pi - diff_phi31)

        df_periodic['FOURIER_DIST'] = np.sqrt(
            10 * (df_periodic['R21'] - R21)**2 + 
            (wrap_phi21)**2 +
            10 * (df_periodic['R31'] - R31)**2 + 
            (wrap_phi31)**2
        )

        best_matches = df_periodic.sort_values('FOURIER_DIST').head(TOP_N)
        with open("config.toml", "a", encoding="utf-8") as config:
            for i in range(len(best_matches)):
                match = best_matches.iloc[i]
                config.write(f"\n[[SAVED_STAR_PARAMS]] # TARGET: {star_label} | TOP {i} | DIST: {match['FOURIER_DIST']}\n") 
                for j in param_arr:
                    config.write(f"{j} = {match[j]}\n")

        print(f"=== TOP {TOP_N} NEAREST NEIGHBOURS ===\n")
        print(best_matches[['alpha', 'mu', 'gamma', 'p', 's', 'R21', 'phi21', 'FOURIER_DIST']], "\n")
        print(f"{TOP_N} NEAREST NEIGHBOUR PARAMS SAVED TO CONFIG\n")


def parse_log_file(filepath="sim_results_space.txt", FILTER_DIVERGENT=True):
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
        
            R21_match = re.search(r"R21:\s*([^\s\|]+)", line)
            phi21_match = re.search(r"phi21:\s*([^\s\|]+)", line)
            
            R31_match = re.search(r"R31:\s*([^\s\|]+)", line)
            phi31_match = re.search(r"phi31:\s*([^\s\|]+)", line)

            if state_match and params_match:
                state = state_match.group(1)
                params_str = params_match.group(1)
                try:
                    params_dict = ast.literal_eval(params_str)
                    params_dict['State'] = state
                    if R21_match and phi21_match:
                        R21_val = R21_match.group(1)
                        phi21_val = phi21_match.group(1)
                        R31_val = R31_match.group(1) if R31_match else "nan"
                        phi31_val = phi31_match.group(1) if phi31_match else "nan"
                        params_dict['R21'] = float(R21_val) if R21_val != "nan" else np.nan
                        params_dict['phi21'] = float(phi21_val) if phi21_val != "nan" else np.nan
                        params_dict['R31'] = float(R31_val) if R31_val != "nan" else np.nan
                        params_dict['phi31'] = float(phi31_val) if phi31_val != "nan" else np.nan
                    else:
                        params_dict['R21'] = np.nan
                        params_dict['phi21'] = np.nan
                    data.append(params_dict)
                except Exception as e:
                    print(f"ERROR {e}. Failed to parse line: {line.strip()} ")
                    
    df = pd.DataFrame(data)
    original_len = len(df)
    if FILTER_DIVERGENT:
        df = df[df['State'] != 'DIVERGENT']
        # df = df[df['State'] != 'STABLE']
        # df = df[df['State'] != 'PERIODIC']
        print(f"SUCCESS: LOADED {original_len} SIMULATION RUNS - KEPT {len(df)} NON-DIVERGENT RUNS")
    else:
        print(f"SUCCESS: LOADED {original_len} SIMULATION RUNS - KEPT {len(df)} NON-DIVERGENT RUNS")
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
    FILTER_DIVERGENT = True

    # UMAP PARAMETERS:
    N_NEIGHBORS  = config.get("N_NEIGHBORS", 15) 
    MIN_DIST     = config.get("MIN_DIST", 0.1)  
    RANDOM_STATE = config.get("RANDOM_STATE", 1)

    # PLOT PARAMETERS
    PLOT_PARAM_SPACE = True
    PLOT_FOURIER_SPACE = True

    # FOURIER SPACE PARAMETERS
    TOP_NEIGH = 1 
    MODEL_LABEL = 'Tanaka-Takeuti Model'
    TARGET_STARS = [{'STAR_LABEL': 'OGLE-LMC-RRLYR-00002', 'R21': 0.447, 'phi21': 4.738, 'R31': 0.206, 'phi31': 3.168}
        #,{'STAR_LABEL': 'OGLE-LMC-RRLYR-00029', 'R21': 0.101, 'phi21': 0.098}
        #,{'STAR_LABEL': 'OGLE-LMC-RRLYR-00001', 'R21': 0.545, 'phi21': 4.395}
        ]
    


    dataset = parse_log_file(FILENAME, FILTER_DIVERGENT=FILTER_DIVERGENT)
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
        for target_star in TARGET_STARS:
            print(f"PROCESSING TARGET STAR: {target_star['STAR_LABEL']}")
            com.plot_fourier_space(dataset=dataset
                            , target_R21=target_star['R21']
                            , target_phi21=target_star['phi21']
                            , model_label=MODEL_LABEL
                            , star_label=target_star['STAR_LABEL']
                            )

            print_and_save_neighbours(dataset=dataset
                                    , param_arr=param_arr
                                    , star_label=target_star['STAR_LABEL']
                                    , R21=target_star['R21']
                                    , phi21=target_star['phi21']
                                    , R31=target_star.get('R31', 0)
                                    , phi31=target_star.get('phi31', 0)
                                    , TOP_N=TOP_NEIGH
                                    )


if __name__ == "__main__": main()
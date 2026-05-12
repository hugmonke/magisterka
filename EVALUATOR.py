import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
import COMMON as com
import tomllib
import re



def parse_ogle(txt):
    matches = re.findall(r'(\w+)\s*=\s*([0-9\.eE\-]+)', txt)
    features = {}
    for k, v in matches:
        clean_key = k.replace('A_1', 'A1').replace('_1', '')
        features[clean_key] = float(v)
    return features

def get_theoretical_trajectory(phases, A1, R21, phi21, R31, phi31):
    """Generates the theoretical OGLE light curve from Fourier features.
    
    Constructs a continuous periodic curve based on the fundamental 
    amplitude and the relative harmonic ratios/phases.
    """
    A2 = R21 * A1
    A3 = R31 * A1
    
    # Standard Fourier series assuming fundamental phase is 0
    y = (A1 * np.cos(2 * np.pi * phases) + 
         A2 * np.cos(4 * np.pi * phases + phi21) + 
         A3 * np.cos(6 * np.pi * phases + phi31))
    return y

def align_generated_trajectory(t, x, dt, target_A1):
    """Phase-folds and scales the raw simulation to match OGLE targets.
    
    Calculates the exact period and phase offset of the simulated trajectory,
    folds the time-series into phase space [0, 1), aligns the primary peak, 
    and scales the overall amplitude to match the physical target.
    """
    
    N = len(x)
    x_centered = x - np.mean(x)
    
    fft_vals = rfft(x_centered)
    amps = np.abs(fft_vals) / N
    phases_fft = np.angle(fft_vals)
    freqs = rfftfreq(N, dt)

    idx_1 = np.argmax(amps[1:]) + 1
    f_1 = freqs[idx_1]
    A_1_sim = amps[idx_1]
    phi_1_sim = phases_fft[idx_1]
    
    if A_1_sim < 1e-10:
        return None, None
        
    folded_phases = (f_1 * t + phi_1_sim / (2 * np.pi)) % 1.0
    scale_factor = target_A1 / A_1_sim
    x_scaled = x_centered * scale_factor
    
    sort_idx = np.argsort(folded_phases)
    return folded_phases[sort_idx], x_scaled[sort_idx]

def main():
    
    
    STAR_LABEL = "OGLE-LMC-RRLYR-00002"
    print(f"STARFIT TRAJECTORY EVALUATOR FOR {STAR_LABEL}\n")
    ogle_data = "A_1 = 0.311 mag, R21_1 = 0.447, phi21_1 = 4.738 rad, R31_1 = 0.206, phi31_1 = 3.168 rad"
    ogle_features = parse_ogle(ogle_data)


    try:
        with open("found_trajectories.toml", "rb") as trajectories:
            raw_trajectories = tomllib.load(trajectories)
            raw_params = raw_trajectories.get(f"{STAR_LABEL}", [])
    except FileNotFoundError as e:
        print(f"{e}: Couldn't load parameteres.")
        exit()

    try:
        with open("config.toml", "rb") as conf:
            config = tomllib.load(conf)
    except FileNotFoundError as e:
        print(f"{e}: config.toml not found.")
        exit()

    DT              = config.get("DT", 0.01)
    T_SKIP          = config.get("T_SKIP", 100)
    T_END           = config.get("T_END", 1000)
    CUTOFF          = config.get("CUTOFF", 1e6)

    FIT_TOLERANCE   = 0.001
    T_SIM           = np.arange(0, T_END, DT)

    params          = raw_params[0]
    print(f"Trajectory parameters: {params}")


    is_valid, x, _, _ = com.get_trajectory_numba(init_xyz=(0.1, 0.0, 0.0)
                                                                   , alpha=params['alpha']
                                                                   , mu=params['mu']
                                                                   , gamma=params['gamma']
                                                                   , p=params['p']
                                                                   , s=params['s']
                                                                   , dt=DT
                                                                   , t_skip=T_SKIP
                                                                   , t_end=T_END
                                                                   , cutoff=CUTOFF
                                                                   )
    if not is_valid:
        print("ERROR: The provided parameters physically diverged.")
        return
        
    sim_phases, sim_scaled = align_generated_trajectory(T_SIM, x, DT, ogle_features['A1'])
    if sim_phases is None:
        print("ERROR: Trajectory has zero amplitude (completely damped).")
        return
    
    plot_phases = np.linspace(0, 1, 1000)
    ogle_theoretical = get_theoretical_trajectory(plot_phases
                                                , ogle_features['A1']
                                                , ogle_features['R21']
                                                , ogle_features['phi21']
                                                , ogle_features.get('R31', 0)
                                                , ogle_features.get('phi31', 0))
    
    sim_interp_func = interp1d(sim_phases, sim_scaled, kind='linear', fill_value="extrapolate")
    sim_interpolated = sim_interp_func(plot_phases)
    mse = np.mean((ogle_theoretical - sim_interpolated)**2)

    print(f">>> Mean Squared Error: {mse}")
    if mse < FIT_TOLERANCE:
        print("> RESULT: GOOD FIT")
    elif mse < 5*FIT_TOLERANCE:
        print("> RESULT: MID FIT")
    else:
        print("> RESULT: BAD FIT")

    plt.figure(figsize=(12, 6))
    two_cycles_phases = np.concatenate([plot_phases, plot_phases + 1])
    two_cycles_ogle = np.concatenate([ogle_theoretical, ogle_theoretical])
    two_cycles_sim_p = np.concatenate([sim_phases, sim_phases + 1])
    two_cycles_sim_x = np.concatenate([sim_scaled, sim_scaled])

    plt.plot(two_cycles_phases, two_cycles_ogle, 'k--', lw=2, label="OGLE Theoretical Fourier Fit")
    plt.plot(two_cycles_sim_p, two_cycles_sim_x, 'r-', alpha=0.75, lw=2, label=f"Tanaka-Takeuti Simulation (MSE: {mse})")
             
    plt.title(f"Simulated Trajectory Overlay: {STAR_LABEL}", fontsize=14)
    plt.xlabel(r"Phase ($\Phi$)", fontsize=12)
    plt.ylabel(r"Relative Amplitude (Scaled to Empirical $A_1$)", fontsize=12)
    plt.gca().invert_yaxis()
    
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
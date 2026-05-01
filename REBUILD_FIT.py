import numpy as np
from scipy.fft import rfft, rfftfreq
import ast
import re
import COMMON as com # Make sure your COMMON.py is in the same folder

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
# FOURIER FEATURES (Your exact function)
# ------------------------------------------------------------
def get_fourier_features(x_array, dt):
    """Performs FFT on the simulated trajectory to extract R21 and phi21."""
    N = len(x_array)
    x_centered = x_array - np.mean(x_array) # center wave
    
    # Optional: If you want to use the Hanning window we discussed, uncomment this:
    # x_centered = x_centered * np.hanning(N)

    fft_vals = rfft(x_centered)
    amps = np.abs(fft_vals)/N
    phases = np.angle(fft_vals)
    freqs = rfftfreq(N, dt)
    
    # FUNDAMENTAL FREQUENCY
    idx_1 = np.argmax(amps[1:]) + 1
    f_1 = freqs[idx_1]
    A_1 = amps[idx_1]
    phi_1 = phases[idx_1]
    
    if A_1 < 1e-10: return None
        
    # HARMONIC
    idx_2 = np.argmin(np.abs(freqs - 2*f_1))
    A_2 = amps[idx_2]
    phi_2 = phases[idx_2]
    
    R21 = A_2 / A_1
    phi21 = (phi_2 - 2*phi_1) % (2*np.pi)
    
    idx_3 = np.argmin(np.abs(freqs - 3*f_1))
    A_3 = amps[idx_3]
    phi_3 = phases[idx_3]
    
    R31 = A_3 / A_1
    phi31 = (phi_3 - 3*phi_1) % (2*np.pi)
    
    return {"R21": R21, "phi21": phi21, "R31": R31, "phi31": phi31}

# ------------------------------------------------------------
# FILE PROCESSOR
# ------------------------------------------------------------
def rebuild_log_file(input_filename, output_filename, dt=0.01):
    print(f"Reading from {input_filename}...")
    
    lines_processed = 0
    lines_updated = 0
    
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                outfile.write(line)
                continue
            
            lines_processed += 1
            
            # 1. Extract the dictionary of parameters
            params_match = re.search(r"Params:\s*(\{.*?\})", line)
            if not params_match:
                outfile.write(line)
                continue
                
            params = ast.literal_eval(params_match.group(1))
            
            # 2. Re-simulate the trajectory to get the wave data
            # 2. Re-simulate the trajectory to get the wave data
            try:
                # FIX 1: Pass 'params' directly without the '**' asterisks
                # Added cutoff=1e6 to make sure we don't abort chaotic runs
                x_array, y_array, z_array = get_trajectory(params, dt=dt, t_skip=500, t_end=1000, cutoff=1e6)
                
                # FIX 2: Check if it actually survived before running FFT
                if x_array is not None:
                    fourier = get_fourier_features(x_array, dt)
                else:
                    fourier = None
                    
            except Exception as e:
                # FIX 3: Stop hiding the error! Kill the script and print it.
                print(f"\nCRASH REASON: {repr(e)}")
                print(f"Failed on parameters: {params}")
                exit()
            
            # 4. Replace the old NaN values using Regex substitution
            if fourier is not None:
                r21_str = f"{fourier['R21']:.4f}"
                phi21_str = f"{fourier['phi21']:.4f}"
                r31_str = f"{fourier['R31']:.4f}"
                phi31_str = f"{fourier['phi31']:.4f}"
                lines_updated += 1
            else:
                r21_str = "   nan"
                phi21_str = "   nan"
                r31_str = "   nan"
                phi31_str = "   nan"
            
            # Sub out the old values for the new ones
            new_line = re.sub(r"R21:\s*[^\s\|]+", f"R21: {r21_str:>6}", line)
            new_line = re.sub(r"phi21:\s*[^\s\|]+", f"phi21: {phi21_str:>6}", new_line)
            new_line = re.sub(r"R31:\s*[^\s\|]+", f"R31: {r31_str:>6}", new_line)
            new_line = re.sub(r"phi31:\s*[^\s\|]+", f"phi31: {phi31_str:>6}", new_line)
            
            outfile.write(new_line)

    print(f"\nDone! Processed {lines_processed} lines.")
    print(f"Successfully calculated Fourier features for {lines_updated} trajectories.")
    print(f"Saved results to: {output_filename}")

if __name__ == "__main__":
    # Point this to your actual log file
    INPUT_FILE = "sim_results_space.txt"
    OUTPUT_FILE = "sim_results_space_FIXED.txt"
    
    rebuild_log_file(INPUT_FILE, OUTPUT_FILE)
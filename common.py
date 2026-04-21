import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
import warnings
from numba import njit
from sklearn.preprocessing import StandardScaler
import umap
from scipy.fft import rfft, rfftfreq
# ------------------------------------------------------------
# RANDOM PARAMETERS
# ------------------------------------------------------------
def get_parameteres(params: dict = None, size: int = 1000):
    """Gets model parameters. Generates them if params is None.

    Args:
        params (dict): Model parameters containing keys:
            - "alpha"
            - "mu"
            - "gamma"
            - "p"
            - "s" 
        Defaults to None.

    Returns:
        dict: Model parameteres.
    """
    if params == None or params == {}:
        params = {
                "alpha": np.random.uniform(-2, 2, size=size)
                ,"mu": np.random.uniform(-1, 1, size=size)
                ,"gamma": np.random.uniform(0.1, 3, size=size)
                ,"p": np.random.uniform(0, 5, size=size)
                ,"s": np.random.uniform(0.5, 2, size=size)
                }
    else:
        params = {param: np.array([val]) for param, val in params.items()}
    return params


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
def get_derivatives(x: np.array, y: np.array, z: np.array, alpha: float, mu: float, gamma: float, p: float, s: float):
    """Gets model derivatives.

    Args:
        x (np.array): Displacement x(t).
        y (np.array): Velocity y(t).
        z (np.array): Force acting on the oscillator z(t).
        - "alpha"
        - "mu"
        - "gamma"
        - "p"
        - "s"

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - dxdt: Time derivative of displacement (dx/dt)
            - dydt: Time derivative of velocity (dy/dt)
            - dzdt: Time derivative of force (dz/dt)
    """

    #         Jacobian
    # |0         1            0|
    # |alfa      mu           1|
    # |0     -(beta + q)     -p|
    # beta and q are lineary dependent -> we merge them into one parameter for
    dxdt = y
    dydt = alpha*x + mu*y + z
    dzdt = -gamma*y - p*z + s*y*z
    return dxdt, dydt, dzdt


# ------------------------------------------------------------
# RK4 SOLVER
# ------------------------------------------------------------
#@njit
def runge_kutta(x: float = 0.1, y: float = 0.0, z: float = 0.0, dt: float = 0.01, alpha=0.0, mu=0.0, gamma=0.0, p=0.0, s=0.0):
    """Gets x, y, z at next timestep.

    Args:
        x (float): Displacement x(t). Defaults to 0.1.
        y (float): Velocity y(t). Defaults to 0.0.
        z (float): Force acting on the oscillator z(t). Defaults to 0.0.
        params (dict): Model parameters containing keys:
            - "alpha"
            - "mu"
            - "gamma"
            - "p"
            - "s"
            Defaults to None.
        dt (float): Timestep.

    Returns:
        tuple[float, float, float]:
            - x_next: Displacement x(t+dt).
            - y_next: Velocity y(t+dt).
            - z_next: Force acting on the oscillator z(t+dt).
    """
    k1x, k1y, k1z = get_derivatives(x, y, z, alpha, mu, gamma, p, s)
    k2x, k2y, k2z = get_derivatives(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z, alpha, mu, gamma, p, s)
    k3x, k3y, k3z = get_derivatives(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z, alpha, mu, gamma, p, s)
    k4x, k4y, k4z = get_derivatives(x + dt*k3x, y + dt*k3y, z + dt*k3z, alpha, mu, gamma, p, s)

    x_next = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    y_next = y + dt*(k1y + 2*k2y + 2*k3y + k4y)/6
    z_next = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    
    return x_next, y_next, z_next



# ------------------------------------------------------------
# ENTROPY
# ------------------------------------------------------------
def shannon_entropy(poinc_x: np.array, poinc_y: np.array, bins: int = 50, floor: int = 10):
    """Calculates entropy of points crossing the Poincare map.

    Args:
        poinc_x (np.array): X dimension of Poincare map.
        poinc_y (np.array): Y dimension of Poincare map.
        bins (int, optional): Number of bins in the 2D histogram. Defaults to 50.
        floor (int, optional): Number of points it requires for entropy to be calculated. Defaults to 10.

    Returns:
        array_like: Shannon entropy.
    """
    if len(poinc_x) < floor: 
        return -1
    
    H, _, _ = np.histogram2d(poinc_x, poinc_y, bins=bins)
    P = H / np.sum(H)
    P = P[P > 0]
    return -np.sum(P * np.log(P)) / np.log(bins * bins)

# ------------------------------------------------------------
# FOURIER FEATURES
# ------------------------------------------------------------
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
# LYAPUNOV EXPONENT AND MODEL SOLUTION
# ------------------------------------------------------------
def solve_and_get_lle(init_xyz: tuple = (0.1, 0.0, 0.0), params: dict = None, dt: float = 0.01, t_skip: int = 50, t_end: int = 150, size: int = 1000, cutoff: int = 1e6):
    """Returns Largest Lyapunov Exponent (LLE).

    Args:
        params (dict): Model parameters. Defaults to None.
        dt (float, optional): Timestep. Defaults to 0.01.
        t_skip (int, optional): Skips calculations until t_skip. Defaults to 50.
        t_end (int, optional): Ends calculations at t_end. Defaults to 150.

    Returns:
        float: Lyapunov exponent.
    """
    if params is None:
        raise ValueError("params cannot be None")
    N_skip = int(t_skip / dt)
    N_sim = int(t_end / dt)

    x_matrix, y_matrix, z_matrix = np.zeros((N_sim, size)), np.zeros((N_sim, size)), np.zeros((N_sim, size))
    epsilon = 1e-8
    x0, y0, z0 = np.full(size, init_xyz[0], dtype=float), np.full(size, init_xyz[1], dtype=float), np.full(size, init_xyz[2], dtype=float)

    alpha, mu = params["alpha"], params["mu"]
    gamma, p, s = params["gamma"], params["p"], params["s"]

    with np.errstate(all='ignore'):
        for _ in range(N_skip):
            x0, y0, z0 = runge_kutta(x=x0, y=y0, z=z0, dt=dt, alpha=alpha, mu=mu, gamma=gamma, p=p, s=s)

        x1, y1, z1 = x0 + epsilon, y0, z0
        d0 = epsilon
        sum_log = np.zeros(size)
        valid_mask = np.ones(size, dtype=bool)
        
        for i in range(N_sim):
            x0, y0, z0 = runge_kutta(x=x0, y=y0, z=z0, dt=dt, alpha=alpha, mu=mu, gamma=gamma, p=p, s=s)
            x_matrix[i, :], y_matrix[i, :], z_matrix[i, :] = x0, y0, z0
            is_safe = np.logical_and.reduce([np.abs(x0) < cutoff, np.abs(y0) < cutoff, np.logical_not(np.isnan(x0))])
            valid_mask = np.logical_and(valid_mask, is_safe)
            
            x1, y1, z1 = runge_kutta(x=x1, y=y1, z=z1, dt=dt, alpha=alpha, mu=mu, gamma=gamma, p=p, s=s)
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            safe_d = np.where(np.logical_or.reduce([np.isnan(d), np.isinf(d)]), 1.0, d)
            safe_d = np.where(safe_d == 0, 1e-16, safe_d)

            sum_log += np.where(valid_mask, np.log(safe_d / d0), 0.0)
            scale = np.where(safe_d > 0, d0/safe_d, 0.0)

            x1 = x0 + dx * scale 
            y1 = y0 + dy * scale 
            z1 = z0 + dz * scale

    return sum_log / (N_sim * dt), valid_mask, x_matrix, y_matrix, z_matrix


# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------
def classify(entropy: float, lle: float):
    """Classifies the model's Dynamical Regime

    Args:
        entropy (float): Shannon entropy.
        lle (float): Lyapunov exponent.

    Returns:
        str: Model's Dynamical Regime.
    """
    # STABLE: LLE is definitively negative. 
    # (We ignore Entropy as it might be inflated by transient spirals)
    if lle < -0.01:
        return "STABLE"
        
    # Crossed the plane less than 10 times
    elif entropy == -1:
        # DIVERGENT: If LLE is mathematically flat, it is flying off in a straight line
        if lle < 0.001:
            return "DIVERGENT"
        # QUASI PERIODIC: If LLE is slightly positive, its probably quasi periodic
        elif lle < 0.05:
            return "QUASI_PERIODIC" 
        else:
        # CHAOTIC: If LLE very positive and there is some entropy
            return "CHAOTIC"
            
    # PERIODIC: LLE is near zero, but enough points for entropy (>0)
    elif lle < 0.005:
        return "PERIODIC"
        
    # QUASI-PERIODIC: If LLE is slightly positive, 
    # but the Poincare map shows big entropy (e.g due to a big ring)
    elif lle < 0.05:
        if entropy > 0.4:
            return "QUASI_PERIODIC"
        else:
            # PERIODIC: If LLE is slightly positive, but entropy is low
            return "PERIODIC"
            
    # CHAOTIC: If LLE is very positive
    else:
        return "CHAOTIC"
    
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
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, marker='s', s=1, picker=True, pickradius=5)

    for state_type, color in color_map.items():
        if state_type in df_states.values:
            ax.scatter([], [], c=color, label=state_type, s=20)

    def on_pick(event):
        idx = event.ind[0] # Picking 1st element solves overlapping points
        param = df_params.iloc[idx]  # parameters and classified state
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

def plot_fourier_space(dataset, target_R21, target_phi21, model_label, star_label):
    """Plots the Accessible Fourier Space vs the Target Observational Star."""
    print("=== Plotting Accessible Fourier Space ===")

    df_periodic = dataset[(dataset['State'] == 'PERIODIC') & (dataset['R21'].notna())]
    if df_periodic.empty:
        print("MAP_MAKER.py: df_periodic is empty - No valid Fourier features")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_periodic['phi21'], df_periodic['R21'], c='blue', alpha=0.3, s=10, label=model_label) # theoretically possible points
    ax.scatter(target_phi21, target_R21, c='red', marker='*', s=100, edgecolor='black', label=star_label) # Target Star
    
    plt.title("Fourier Space with Target Star")
    plt.xlabel(r"Phase Difference $\phi_{21}$ (radians)")
    plt.ylabel(r"Amplitude Ratio $R_{21}$")
    plt.xlim(0, 2*np.pi)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
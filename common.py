import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from sklearn.preprocessing import StandardScaler
import umap
from scipy.fft import rfft, rfftfreq

def get_parameteres(params: dict = None, size: int = 1000):
    """Gets model parameters as numpy arrays. Generates them if params is None.

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
                "alpha": np.random.uniform(-2, 10, size=size)
                ,"mu": np.random.uniform(-1, 10, size=size)
                ,"gamma": np.random.uniform(0.1, 5, size=size)
                ,"p": np.random.uniform(0, 5, size=size)
                ,"s": np.random.uniform(0, 5, size=size)
                }
    else:
        params = {param: np.array([val]) for param, val in params.items()}
    return params


def generate_plane(point, normal):
    """Calculates the coefficients of a plane equation (ax + by + cz + d = 0).

    Takes a point on the plane and a normal vector, normalizes the normal vector to a unit length of 1, 
    and calculates the scalar distance component (d) from the origin.

    Args:
        point (array-like): A 3D coordinate (x, y, z) specifying a point that the plane passes through.
        normal (array-like): A 3D vector (nx, ny, nz) perpendicular to the plane.

    Returns:
        tuple: A 4-element tuple (a, b, c, d) of floats representing the coefficients of the plane equation, 
        where (a, b, c) is the normalized normal vector.
    """

    point = np.array(point)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # normalize
    a, b, c = normal
    d = -np.dot(normal.T, point.T)

    return a, b, c, d


def poincare_map(x, y, z, plane):
    """Calculates the Poincaré section of a 3D trajectory intersecting a plane.

    Iterates through a given 3D trajectory and identifies points where the trajectory
    crosses a specified plane in phase space. Uses linear interpolation between discrete time steps to 
    approximate the exact coordinates of the intersection.

    Args:
        x (array-like): X-coordinates representing the trajectory.
        y (array-like): Y-coordinates representing the trajectory.
        z (array-like): Z-coordinates representing the trajectory.
        plane (tuple): Tuple representing the coefficients of the intersecting plane equation (ax + by + cz + d = 0).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - poinc_x: X-coordinates of all calculated intersection points on the Poincaré section.
            - poinc_y: Y-coordinates of all calculated intersection points on the Poincaré section.
            - poinc_z: Z-coordinates of all calculated intersection points on the Poincaré section.
    """
    a, b, c, d = plane
    def plane_eq(x, y, z): 
        return a*x + b*y + c*z + d
    poinc_x = []
    poinc_y = [] 
    poinc_z = []

    for i in range(1, len(x)):
        f_prev = plane_eq(x[i-1], y[i-1], z[i-1])
        f_curr = plane_eq(x[i], y[i], z[i])

        # count BOTH upward and downward crossings
        if (f_prev < 0 and f_curr >= 0) or (f_prev > 0 and f_curr <= 0):
            t_cross = abs(f_prev)/(abs(f_prev)+abs(f_curr))
            xi = x[i-1] + t_cross*(x[i]-x[i-1])
            yi = y[i-1] + t_cross*(y[i]-y[i-1])
            zi = z[i-1] + t_cross*(z[i]-z[i-1])
            poinc_x.append(xi)
            poinc_y.append(yi)
            poinc_z.append(zi)

    return np.array(poinc_x), np.array(poinc_y), np.array(poinc_z)

@njit
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


@njit
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


def shannon_entropy(poinc_x: np.array, poinc_y: np.array, bins: int = 200, floor: int = 10):
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

def get_fourier_features(x_array, dt):
    """Extracts Fourier amplitude ratios and phase differences from a time series.

    Performs a real Fast Fourier Transform on a 1D trajectory to isolate its fundamental frequency and harmonics. 
    It calculates the amplitude ratios (R_k1 = A_k / A_1) and phase differences (phi_k1 = phi_k - k*phi_1) 
    for the second and third harmonics. The wave is centered prior to the FFT to remove any DC offset.

    Args:
        x_array (array-like): Trajectory data.
        dt (float): Time step between consecutive points in x_array.

    Returns:
        dict or None: Dictionary containing the extracted Fourier parameters:
            - 'R21' (float): Amplitude ratio of the 2nd harmonic to the fundamental.
            - 'phi21' (float): Phase difference of the 2nd harmonic (mod 2π).
            - 'R31' (float): Amplitude ratio of the 3rd harmonic to the fundamental.
            - 'phi31' (float): Phase difference of the 3rd harmonic (mod 2π).
            
            Returns 'None' if the fundamental amplitude < 1e-10, which indicates a flat or completely damped signal.
    """

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

def classify(entropy: float, lle: float):
    """Classifies the model's Dynamical Regime

    Args:
        entropy (float): Shannon entropy.
        lle (float): Lyapunov exponent.

    Returns:
        str: Model's Dynamical Regime.
    """
    if lle < -0.01:
        return "STABLE"

    if entropy == -1 or (entropy > 0.45 and lle < 0.03):
        return "DIVERGENT"

    if lle > 0.05 and entropy > 0.35:
        return "CHAOTIC"

    else: # lle < 0.01
        if entropy < 0.35:
            return "PERIODIC"
        else:
            return "QUASI_PERIODIC"
    
    import numpy as np

def validate_state_and_features(x_array, dt, state, tolerance=0.05):
    """Validates trajectory stability and extracts Fourier parameters.

    Acts as a secondary filter for numerical stability. Splits the time series in half and checks for unphysical baseline drift or amplitude growth
    for trajectories classified as potentially stable limit cycles or strange attractors (CHAOTIC, QUASI_PERIODIC, PERIODIC).
    Calculates and returns the Fourier features if the trajectory is stable within the specified tolerance.

    Args:
        x_array (array-like): 1D trajectory data.
        dt (float): Integration time step.
        initial_state (str): Preliminary dynamical classification.
        tolerance (float, optional): Max allowed fractional mean or amplitude change. Defaults to 0.05.

    Returns:
        tuple: A 2-element tuple '(final_state, features):
            - final_state (string): Updated classification string
            - features (dict): Contains 'R21', 'phi21', 'R31', and 'phi31' (np.nan if state is divergent).
    """

    final_state = state
    features = {"R21": np.nan, "phi21": np.nan, "R31": np.nan, "phi31": np.nan}

    if state in ["CHAOTIC", "QUASI_PERIODIC", "PERIODIC"]:
        half_idx = len(x_array) // 2
        
        amplitude_total = np.max(x_array) - np.min(x_array) + 1e-9
        mean_diff = abs(np.mean(x_array[:half_idx]) - np.mean(x_array[half_idx:]))
        drift_ratio = mean_diff / amplitude_total

        amp_start = np.max(x_array[:half_idx]) - np.min(x_array[:half_idx])
        amp_end = np.max(x_array[half_idx:]) - np.min(x_array[half_idx:])
        amp_growth = abs(amp_end - amp_start) / (amp_start + 1e-9)

        extracted_features = get_fourier_features(x_array, dt)
        if extracted_features is not None:
            features = extracted_features

            if drift_ratio >= tolerance or amp_growth >= tolerance:
                final_state = "DIVERGENT"
        else:
            final_state = "DIVERGENT"
                
    return final_state, features
    
def plot_parameter_space(df_params, df_states, n_neighbors, min_dist, random_state):
    """UMAP dimensionality reduction and an interactive 2D projection generation.

    Scales the input parameter space using standard scaling, reduces it to two dimensions using UMAP, 
    and creates a scatter plot colored by the dynamical state of each point. 
    It includes an interactive pick event: clicking a point on the plot will print its exact parameters and append them to a local 'config.toml' file for later use.

    Args:
        df_params (pd.DataFrame): DataFrame containing input parameters for the simulations.
        df_states (pd.Series): Series containing classified dynamical states corresponding to each parameter set.
        n_neighbors (int):  UMAP local neighborhood size.
        min_dist (float): UMAP point packing density
        random_state (int): Random seed for reproducible UMAP projections.

    Returns:
        None: The function opens a matplotlib display window and does not return a value.
    """
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



def plot_fourier_space(dataset, target_R21, target_phi21, target_R31, target_phi31, model_label, star_label, n_neighbors=15, min_dist=0.1):
    """Projects 4D accessible Fourier space down to 2D using UMAP.

    Filters the provided dataset for valid 4D Fourier features (R21, phi21, R31, phi31).
    Standardizes the features and uses UMAP to generate a 2D topological mapping. 
    The empirical target star is transformed using the same fitted UMAP model to 
    accurately display its mathematical distance from the simulated points.

    
    TODO: It calculates UMAP anew for every star; its not needed!! We should cache this for next stars
    Args:
        dataset (pd.DataFrame): DataFrame containing the simulation results.
        target_R21 (float): Target amplitude ratio R21.
        target_phi21 (float): Target phase difference phi21 (radians).
        target_R31 (float): Target amplitude ratio R31.
        target_phi31 (float): Target phase difference phi31 (radians).
        model_label (str): Legend label for simulated points.
        star_label (str): Legend label for the target star.
        n_neighbors (int, optional): UMAP local neighborhood size. Defaults to 15.
        min_dist (float, optional): UMAP point packing density. Defaults to 0.1.

    Returns:
        None: The function opens a matplotlib display window and does not return a value.
    """

    print("=== Plotting UMAP 4D Fourier Space ===")

    features = ['R21', 'phi21', 'R31', 'phi31']
    df_filtered = dataset.dropna(subset=features).copy()
    
    if df_filtered.empty:
        print("MAP_MAKER.py: df_filtered is empty - No valid 4D Fourier features")
        return

    scaler = StandardScaler()
    X = df_filtered[features].values
    X_target = np.array([[target_R21, target_phi21, target_R31, target_phi31]])

    X_scaled = scaler.fit_transform(X)
    X_target_scaled = scaler.transform(X_target)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    target_umap = reducer.transform(X_target_scaled) 

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c='blue', alpha=0.3, s=10, label=model_label)
    
    ax.scatter(target_umap[:, 0], target_umap[:, 1], c='red', marker='*', s=50, edgecolor='black', zorder=5, label=star_label)
    
    plt.title("UMAP Fourier Space 2D Projection ")
    plt.xlabel("DIM 1")
    plt.ylabel("DIM 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
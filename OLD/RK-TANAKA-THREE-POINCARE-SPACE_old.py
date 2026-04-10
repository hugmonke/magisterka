import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# PARAMETERS & MODEL
# ------------------------------------------------------------
def model_parameteres():
    params = {
              "alpha": np.random.uniform(-2, 2)
              ,"mu": np.random.uniform(-1, 1)
              ,"gamma": np.random.uniform(0.1, 3)
              ,"p": np.random.uniform(0, 3)
              ,"s": np.random.uniform(0.5, 2)
            }
    
    return params

def derivatives(x, y, z, params):
    #         Jacobian
    # |0         1            0|
    # |alfa      mu           1|
    # |0     -(beta + q)     -p|
    # beta and q are lineary dependent -> we merge them into one parameter for
    dxdt = y
    dydt = params["alpha"]*x + params["mu"]*y + z
    dzdt = -params["gamma"]*y - params["p"]*z + params["s"]*y*z
    return dxdt, dydt, dzdt

def runge_kutta(x, y, z, param, dt):
    k1x, k1y, k1z = derivatives(x, y, z, param)
    k2x, k2y, k2z = derivatives(x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z, param)
    k3x, k3y, k3z = derivatives(x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z, param)
    k4x, k4y, k4z = derivatives(x + dt*k3x, y + dt*k3y, z + dt*k3z, param)

    x_next = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    y_next = y + dt*(k1y + 2*k2y + 2*k3y + k4y)/6
    z_next = z + dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    
    return x_next, y_next, z_next

# ------------------------------------------------------------
# SIMULATION & DYNAMICAL ANALYSIS
# ------------------------------------------------------------
def solve_system(params, dt=0.01, t_skip=50, t_end=150):

    # We dont care about the simulation until initial t_skip time
    N_skip = int(t_skip / dt)
    N_sim = int(t_end / dt)

    x_arr = np.zeros(N_sim)
    y_arr = np.zeros(N_sim)
    z_arr = np.zeros(N_sim)

    x0 = 0.1
    y0 = 0
    z0 = 0

    for _ in range(N_skip):
        x0, y0, z0 = runge_kutta(x0, y0, z0, params, dt)

    x_arr[0], y_arr[0], z_arr[0] = x0, y0, z0
    for i in range(N_sim-1):
        x_arr[i+1], y_arr[i+1], z_arr[i+1] = runge_kutta(x_arr[i], y_arr[i], z_arr[i], params, dt)

        if np.abs(x_arr[i+1]) > 1e6 or np.abs(y_arr[i+1]) > 1e6:
            raise ValueError("System Diverged")
    return x_arr, y_arr, z_arr

# ------------------------------------------------------------
# ENTROPY
# ------------------------------------------------------------
def shannon_entropy(poinc_x, poinc_y, bins=50, floor=10):

    if len(poinc_x) < floor: 
        # If trajectory crossed plane less than *floor* times, its probably stable
        return 0
    
    H, _, _ = np.histogram2d(poinc_x, poinc_y, bins=bins)
    P = H / np.sum(H)
    P = P[P > 0]
    return -np.sum(P * np.log(P)) / np.log(bins * bins)

# ------------------------------------------------------------
# LYAPUNOV EXPONENT
# ------------------------------------------------------------
def lyapunov_exponent(params, dt=0.01, t_skip=50, t_end=150):
    """Returns Largest Lyapunov Exponent (LLE)."""
    N_skip = int(t_skip / dt)
    N_sim = int(t_end / dt)

    epsilon = 1e-8
    x1 = 0.1
    y1 = 0
    z1 = 0

    for _ in range(N_skip):
        x1, y1, z1 = runge_kutta(x1, y1, z1, params, dt)

    x2 = x1 + epsilon
    y2 = y1
    z2 = z1
    d0 = epsilon
    sum_log = 0

    for _ in range(N_sim):
        x1, y1, z1 = runge_kutta(x1, y1, z1, params, dt)
        x2, y2, z2 = runge_kutta(x2, y2, z2, params, dt)

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if d == 0 or np.isnan(d) or np.isinf(d):
            continue

        sum_log += np.log(d/d0)
        scale = d0/d
        x2 = x1 + dx * scale 
        y2 = y1 + dy * scale 
        z2 = z1 + dz * scale

    return sum_log / (N_sim * dt)

def classify(entropy, lle):
    if lle > 0.01:
        return "CHAOTIC"
    elif entropy < 0.2 and lle < -0.01:
        return "STABLE"
    else:
        return "PERIODIC"

# ------------------------------------------------------------
# DATA GENERATION
# ------------------------------------------------------------
def get_dataset(sim_num=1000, print_every=50):
    print(f"Running {sim_num} simulations to map parameter space")
    overflow_count = 0
    results = []

    for i in range(sim_num):
        params = model_parameteres()
        
        try:
            lle = lyapunov_exponent(params)
            x, y, z = solve_system(params)
            z_mean = np.mean(z)
            # Poincare Map Crosses z=0 for speed
            poinc_x, poinc_y = [], []
            for j in range(1, len(z)):
                if (z[j-1] - z_mean) * (z[j] - z_mean) < 0:
                    poinc_x.append(x[j])
                    poinc_y.append(y[j])
            
            entropy = shannon_entropy(poinc_x, poinc_y)
            state = classify(entropy, lle)
            
            run_data = params.copy()
            run_data.update({"Entropy": entropy, "LLE": lle, "State": state})
            results.append(run_data)

        except (OverflowError, ValueError, FloatingPointError):
            print(f"Overflow for these parameters: {params}")
            overflow_count += 1
            pass

        if (i+1) % print_every == 0:
            print(f"Run no. {i+1} ")

    print(f"Total successful runs: {len(results)}")
    print(f"Total exploded (error) runs: {overflow_count}")
    return pd.DataFrame(results)

# ------------------------------------------------------------
# MACHINE LEARNING & MAPPING (UMAP)
# ------------------------------------------------------------
if __name__ == "__main__":
    np.seterr(all='raise') # Overflow warnings as errors so try block can catch them
    df = get_dataset(sim_num=1000)
    if df.empty or len(df['State'].unique()) < 2:
        print("Not enough varied data. Increase number of simulations (sim_num).")
        exit()

    features = ['alpha', 'mu', 'gamma', 'p', 's']
    X = df[features]
    y = df['State']

    # UMAP for nonlinear Dim Reduction
    print("=== Running UMAP projection ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP parameters:
    # n_neighbors controls how UMAP balances local vs global structure.
    # min_dist controls how tightly points are packed together.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    # Parameter Space (UMAP Projection) Plot 
    plt.figure(figsize=(10, 6))
    color_map = {"CHAOTIC": "red", "STABLE": "green", "PERIODIC": "blue"}
    
    for state_type in color_map.keys():
        if state_type in df['State'].values:
            mask = y == state_type
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                        c=color_map[state_type], label=state_type, 
                        alpha=0.7, edgecolors='w', s=60)

    plt.title("Parameter Space Mapping (UMAP Projection)")
    plt.xlabel("DIM 1")
    plt.ylabel("DIM 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Random Forest for Feature Importance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Random Forest Classification Accuracy: {accuracy*100:.2f}%")
    
    # Feature Importances
    print("=== Feature Importances ===")
    importances = clf.feature_importances_
    
    sorted_indices = np.argsort(importances)[::-1]
    for idx in sorted_indices:
        print(f"{features[idx]:>5}: {importances[idx]*100:.1f}%")
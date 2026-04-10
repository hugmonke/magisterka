import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

# fajne Parameters: {'alpha': -0.5502103593764673, 'mu': 0.6915052784410058, 'beta': 0.5784962839255214, 'p': 4.010755401427479, 'q': 0.5108874980348869, 's': 1.6773485833457182}
# {'alpha': 1.2784327077565871, 'mu': -0.009198275202455308, 'gamma': 1.3066666741719266, 'p': 0.8194350668953788, 's': 0.845652967001836}
# should be stable, is periodic  {'alpha': -1.6545026753188203, 'mu': 0.2663167113878475, 'gamma': 1.8199676577124784, 'p': 2.644263331982729, 's': 1.0641127893623232}
# ------------------------------------------------------------
# RANDOM PARAMETERS
# ------------------------------------------------------------
def get_parameteres(params: dict = None):
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
    if params == None:
        params = {
                "alpha": np.random.uniform(-2, 2)
                ,"mu": np.random.uniform(-1, 1)
                ,"gamma": np.random.uniform(0.1, 3)
                ,"p": np.random.uniform(0, 3)
                ,"s": np.random.uniform(0.5, 2)
                }
    
    return params


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
def get_derivatives(x: float = 0.1, y: float = 0.0, z: float = 0.0, params: dict = None):
    """Gets model derivatives.

    Args:
        x (float): Displacement x(t).
        y (float): Velocity y(t).
        z (float): Force acting on the oscillator z(t).
        params (dict): Model parameters containing keys:
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
    if params is None:
        raise ValueError("params cant be None")
    #         Jacobian
    # |0         1            0|
    # |alfa      mu           1|
    # |0     -(beta + q)     -p|
    # beta and q are lineary dependent -> we merge them into one parameter for
    dxdt = y
    dydt = params["alpha"]*x + params["mu"]*y + z
    dzdt = -params["gamma"]*y - params["p"]*z + params["s"]*y*z
    return dxdt, dydt, dzdt


# ------------------------------------------------------------
# RK4 SOLVER
# ------------------------------------------------------------
def runge_kutta(x: float = 0.1, y: float = 0.0, z: float = 0.0, params: dict = None, dt: float = 0.01):
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
    k1x, k1y, k1z = get_derivatives(x=x, y=y, z=z, params=params)
    k2x, k2y, k2z = get_derivatives(x=x + 0.5*dt*k1x, y=y+0.5*dt*k1y, z=z+0.5*dt*k1z, params=params)
    k3x, k3y, k3z = get_derivatives(x=x + 0.5*dt*k2x, y=y+0.5*dt*k2y, z=z+0.5*dt*k2z, params=params)
    k4x, k4y, k4z = get_derivatives(x=x+dt*k3x, y=y+dt*k3y, z=z+dt*k3z, params=params)

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
        # If trajectory crossed plane less than *floor* times, its probably stable
        return 0
    
    H, _, _ = np.histogram2d(poinc_x, poinc_y, bins=bins)
    P = H / np.sum(H)
    P = P[P > 0]
    return -np.sum(P * np.log(P)) / np.log(bins * bins)


# ------------------------------------------------------------
# LYAPUNOV EXPONENT AND MODEL SOLUTION
# ------------------------------------------------------------
def solve_and_get_lle(init_xyz: tuple = (0.1, 0.0, 0.0), params: dict = None, dt: float = 0.01, t_skip: int = 50, t_end: int = 50, cutoff: int = 1e6):
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

    x_arr, y_arr, z_arr = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    epsilon = 1e-8
    x0, y0, z0 = init_xyz

    for _ in range(N_skip):
        x0, y0, z0 = runge_kutta(x0, y0, z0, params, dt)

    x1, y1, z1 = x0 + epsilon, y0, z0
    d0 = epsilon
    sum_log = 0

    for i in range(N_sim):
        x0, y0, z0 = runge_kutta(x=x0, y=y0, z=z0, params=params, dt=dt)
        x_arr[i], y_arr[i], z_arr[i] = x0, y0, z0

        if np.abs(x0) > cutoff or np.abs(y0) > cutoff:
            raise ValueError("System Diverged")
        x1, y1, z1 = runge_kutta(x=x1, y=y1, z=z1, params=params, dt=dt)
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if d == 0 or np.isnan(d) or np.isinf(d):
            continue

        sum_log += np.log(d/d0)
        scale = d0/d
        x1 = x0 + dx * scale 
        y1 = y0 + dy * scale 
        z1 = z0 + dz * scale

    return sum_log / (N_sim * dt), x_arr, y_arr, z_arr


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
    if lle > 0.01:
        return "CHAOTIC"
    elif entropy < 0.2 and lle < -0.01:
        return "STABLE"
    else:
        return "PERIODIC"
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
import warnings

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
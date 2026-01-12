"""
Utility functions for data assimilation examples.
"""

import numpy as np


def lorenz_dynamics(t, Z, beta=8/3, rho=28, sigma=10):
    """
    Lorenz-63 dynamics.
    
    Parameters
    ----------
    t : float
        Time (not used, but required for ODE solver interface)
    Z : ndarray
        State vector(s), shape (3,) for single particle or (N, 3) for ensemble
    beta : float
        Lorenz parameter (default: 8/3)
    rho : float
        Lorenz parameter (default: 28)
    sigma : float
        Lorenz parameter (default: 10)
        
    Returns
    -------
    dyn : ndarray
        Time derivative of state vector(s)
    """
    if len(Z.shape) == 1:  # Only one particle
        dZ1ds = -sigma * Z[0] + sigma * Z[1]
        dZ2ds = -Z[0] * Z[2] + rho * Z[0] - Z[1]
        dZ3ds = Z[0] * Z[1] - beta * Z[2]
        
        dyn = np.asarray([dZ1ds, dZ2ds, dZ3ds])
    else:
        dZ1ds = -sigma * Z[..., 0] + sigma * Z[..., 1]
        dZ2ds = -Z[..., 0] * Z[..., 2] + rho * Z[..., 0] - Z[..., 1]
        dZ3ds = Z[..., 0] * Z[..., 1] - beta * Z[..., 2]

        dyn = np.column_stack((dZ1ds, dZ2ds, dZ3ds))

    return dyn


def rk4(Z, fun, t=0, dt=1, nt=1):
    """
    Fourth-order Runge-Kutta integration scheme.
    
    Parameters
    ----------
    Z : ndarray
        Initial states
    fun : callable
        Function to be integrated
    t : float
        Initial time (default: 0)
    dt : float
        Time step length (default: 1)
    nt : int
        Number of time steps (default: 1)
        
    Returns
    -------
    Z : ndarray
        Updated states after integration
    """
    # Prepare array for use
    if len(Z.shape) == 1:  # We have only one particle, convert it to correct format
        Z = Z[np.newaxis, :]
        
    # Go through all time steps
    for i in range(nt):
        # Calculate the RK4 values
        k1 = fun(t + i * dt, Z)
        k2 = fun(t + i * dt + 0.5 * dt, Z + dt / 2 * k1)
        k3 = fun(t + i * dt + 0.5 * dt, Z + dt / 2 * k2)
        k4 = fun(t + i * dt + dt, Z + dt * k3)
    
        # Update next value
        Z += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return Z

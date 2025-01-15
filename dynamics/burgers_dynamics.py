import numpy as np

def burgers_update_original(y, u, dx, dt, nu=0.01):
    r"""
    Given current state y and control signal u at some time t, return the updated state vector y at time t+dt.
    The boundary condition is 0.

    :param y: (n,), state vector.
    :param u: (m,), control signal.
    :param dx: the spatial step size.
    :param dt: the temporal step size.
    :param nu: kinematic viscosity.

    :return: (n,), the updated state vector.
    """
    assert type(y) == np.ndarray and type(u) == np.ndarray, "y and u must be numpy arrays"
    assert y.ndim == 1 and u.ndim == 1, "y and u must be 1 dimensional arrays"
    assert y.shape == u.shape, "y and u must have the same shape"
    
    y_new = np.copy(y)
    ya=np.roll(y,-1) # Equivalent to y_{i+1}
    yb=np.roll(y,1) # Equivalent to y_{i-1}
    convective_term = (ya**2/2 - yb**2/2) / (2*dx) # Convective term (advection)
    diffusive_term = nu*(ya - 2*y + yb) / (dx**2) # Diffusive term (viscosity)
    y_new = y_new - dt*convective_term + dt*diffusive_term + u*dt # Lax-Friedrichs update with viscosity
    y_new[0] = 0 # Apply 0 boundary conditions
    y_new[-1] = 0 # Apply 0 boundary conditions
    return y_new

nu=0.01 # Kinematic viscosity
dx=1/128 # Spatial node interval
dt=0.0001 # Tiny time length of a single update of the differential equation


def burgers_update(state, action, delta_t, t):
    """
    Update the system for a time interval of delta_t iteratively, where the action remains unchanged.
    """
    global nu, dx, dt
    y = state
    for i in range(int(delta_t/dt)):
        y = burgers_update_original(y, action, dx=dx, dt=dt, nu=nu)
    return y






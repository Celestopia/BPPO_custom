import numpy as np

g = 9.81
m = 1.0 # mass (kg)
L = 1.0 # length (m)
mu = 0.1 # friction coefficient

def ip_update(state, action, delta_t, t): 
    """
    1D Nonlinear Inverted Pendulum equation.
    state: [theta, theta_dot] (angle, angular velocity)
    u: torque input
    t: current time
    delta_t: time step length
    """
    theta, theta_dot = state
    
    # Dynamics of nonlinear inverted pendulum system
    theta_ddot = (g * np.sin(theta) - mu * theta_dot) / L + action / (m * L**2)
    
    # Update the state using Euler's method
    new_theta = theta + theta_dot * delta_t
    new_theta_dot = theta_dot + theta_ddot * delta_t
    
    return np.array([new_theta.item(), new_theta_dot.item()])
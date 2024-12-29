import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(100)

# Function to initialize u(x, 0) as a Gaussian, now centered around 0
def generate_initial_y(x):
    """
    Generate the initial state u(0, x) as a superposition of two Gaussian functions.
    """
    # Sampling parameters for the Gaussian functions
    a1 = np.random.uniform(0, 2)
    a2 = np.random.uniform(-2, 0)
    mu1 = np.random.uniform(-1, 1)
    mu2 = np.random.uniform(-0.5, 0.5)
    sigma1 = np.random.uniform(0.3, 0.6)
    sigma2 = np.random.uniform(0.4, 0.8)
    
    u = a1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2)) + a2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    u[0] = 0
    u[-1] = 0
    return u


def generate_control_sequence(x, t):
    """
    Generate control sequence w(t, x) as a superposition of 8 Gaussian functions.
    """
    w = np.zeros_like(x)
    for i in range(8):
        ind = np.random.binomial(1, 0.5)
        if i==0:
            ai = np.random.uniform(-1.5,1.5)
        else:
            if ind:
                ai = np.random.uniform(-1.5,1.5)
            else:
                ai = 0
        b1_i = np.random.uniform(0, 1)
        b2_i = np.random.uniform(0, 1)
        sigma1_i = np.random.uniform(0.05, 0.2)
        sigma2_i = np.random.uniform(0.05, 0.2)
        
        w += ai * np.exp(-((x - b1_i)**2) / (2 * sigma1_i**2)) * np.exp(-((t - b2_i)**2) / (2 * sigma2_i**2))
    
    return w

def burgers_update(y, u, dx, dt, nu=0.01):
    '''
    Discretized Burgers equation solver using Lax-Friedrichs method with viscosity
    '''
    y_new = np.copy(y)
    for i in range(1, len(y) - 1): # Loop over the interior points and apply the Lax-Friedrichs scheme with viscosity
        convective_term = (y[i + 1] - y[i - 1]) / (2 * dx) # Convective term (advection)
        diffusive_term = (y[i + 1] - 2 * y[i] + y[i - 1]) / (dx**2) # Diffusive term (viscosity)
        y_new[i] = 0.5 * (y[i+1] + y[i-1]) - (dt/(2*dx))*y[i]*convective_term + nu*dt*diffusive_term + 0.1*u[i]*dt # Lax-Friedrichs update with viscosity
    # Apply 0 boundary conditions
    y_new[0] = 0
    y_new[-1] = 0
    return y_new

# Parameters
x_range=(-5,5)
nt = 500 # Number of time steps
nx = 128 # Number of spatial nodes (grid points)
dt= 0.001 # Temporal interval
dx = (x_range[1]-x_range[0])/nx # Spatial interval
nu = 0.01  # Kinematic viscosity


x = np.linspace(*x_range, nx) # Initialize the spatial grid
y = generate_initial_y(x) # Set the initial condition





#plt.plot(x, y)
#plt.title("Initial condition")
#plt.xlabel("x")
#plt.ylabel("u(x,0)")
#plt.show()
#
#plt.plot(x, burgers_update(y,np.zeros_like(x),dx,dt))
#plt.title("Initial condition")
#plt.xlabel("x")
#plt.ylabel("u(x,0)")
#plt.show()

Y_bar=[]
for t_idx in range(nt):
    u=generate_control_sequence(y,t_idx*dt) # (nx,), Control vector at time t_idx*dt
    y_new=burgers_update(y, u, dx, dt)
    Y_bar.append(y_new)
    y=y_new # Update y
Y_bar=np.array(Y_bar) # (nt, nx)


# Set up the figure and axis for plotting
fig, ax = plt.subplots()
line, = ax.plot(x, y, label="Burgers equation solution")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(*x_range)
ax.set_title("Burgers Equation Evolution")
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
time_text = ax.text(0.5, 0.95, "",
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center')

# Animation update function
def update(frame):
    line.set_ydata(Y_bar[frame])
    time_text.set_text(f'Current Time: {frame * dt:.3f}/{nt*dt:.3f}')  # 更新时间提示
    return line, time_text

# Create the animation
ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=20)
plt.show()
















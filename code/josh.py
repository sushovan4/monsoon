import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt

# Define the Lorenz 63 system
@jit
def lorenz63(t, state, sigma=10.0, rho=28.0, beta=8.0/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def run_integration(T,rho,initial_state):

    t_span = (0.0, T)
    n=200*T
    t_eval = np.linspace(t_span[0], t_span[1], n)
    specific_system = lambda a,b: lorenz63(a,b,rho=rho)
    # Integrate the system
    solution = solve_ivp(specific_system, t_span, initial_state, t_eval=t_eval, vectorized=True)

    # Extract the results
    t = solution.t
    x, y, z = solution.y
    return t,x,y,z


#Run it like this
initial_state = [1.0, 1.0, 1.0]
T=1000
rho=28

t,x,y,z=run_integration(T,rho,initial_state)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
plt.title("Lorenz 63 System")
plt.show()
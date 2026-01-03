# Here we generate the true dynamics of the system using
# a solver (Euler)
import numpy as np
from matplotlib import pyplot as plt

from pendulum_nonlinear import pendulum_dynamics

# params
m = 1.0  # mass
l = 1.0  # length of rod
dt = 0.05  # time-step
g = 9.8  # gravitational acceleration

init_states = np.array(
    [np.random.normal(loc=[0, 2.0], scale=0.1, size=2) for i in range(100)]
)
final_states = []


# simulation function
def run_simulation(init_state: np.ndarray):
    i = 0
    final_q, final_p = 0, 0
    while i <= 80:
        q, p = init_state
        _, dp_dt = pendulum_dynamics(t=dt, state=[q, p], m=m, l=l, g=g)
        p = p + dp_dt * dt
        dq_dt, _ = pendulum_dynamics(t=dt, state=[q, p], m=m, l=l, g=g)
        q = q + dq_dt * dt
        init_state = [q, p]
        i += 1
        if i == 80:
            final_q, final_p = q, p
    return [final_q, final_p]


# run simulation
for init_points in init_states:
    final_states.append(run_simulation(init_points))

final_states = np.array(final_states)

# plotting
plt.scatter(init_states.flatten()[::2], init_states.flatten()[1::2], c="blue")
plt.scatter(final_states.flatten()[::2], final_states.flatten()[1::2], c="red")
plt.title("Initial positions vs. Final positions")
plt.savefig("./FIG4.png")

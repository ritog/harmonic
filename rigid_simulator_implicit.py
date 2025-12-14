from typing import List

from matplotlib import pyplot as plt

from pendulum_nonlinear import pendulum_dynamics

# different initial states
init_a = [0.5, 0]  # small release
init_b = [3.1, 0]  # close to top
init_c = [0, 5.0]  # close to bottom with force

# params
m = 1.0  # mass
l = 1.0  # length of rod
dt = 0.05  # time-step
g = 9.8  # gravitational acceleration


def run_simulation(init_state: List):
    p_vals = []
    q_vals = []
    for i in range(1_000):
        q, p = init_state
        _, dp_dt = pendulum_dynamics(t=dt, state=[q, p], m=m, l=l, g=g)
        p = p + dp_dt * dt
        dq_dt, _ = pendulum_dynamics(t=dt, state=[q, p], m=m, l=l, g=g)
        q = q + dq_dt * dt
        init_state = [q, p]
        q_vals.append(q)
        p_vals.append(p)
    return p_vals, q_vals


a_p_vals, a_q_vals = run_simulation(init_a)
b_p_vals, b_q_vals = run_simulation(init_b)
c_p_vals, c_q_vals = run_simulation(init_c)

# Plotting
plt.plot(a_q_vals, a_p_vals, label="$q_0=0, p_0=0$")
plt.plot(b_q_vals, b_p_vals, label="$q_0=3.1, p_0=0$")
plt.plot(c_q_vals, c_p_vals, label="$q_0=0, p_0=5.0$")

plt.xlabel("$q$")
plt.ylabel("$p$")
plt.title("$p v. q$ for different inital conditions")
plt.legend()
plt.tight_layout()
plt.show()

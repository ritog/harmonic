from matplotlib import pyplot as plt

from harmonic import harmonic_dynamics

# params
m = 1.0  # mass
k = 1.0  # spring constant
q0 = 0  # initial position
p0 = 1  # initial momentum
dt = 0.1  # time-step

init_state = q0, p0
q_vals = []
p_vals = []

for i in range(0, 200):
    q, p = init_state
    dq_dt, dp_dt = harmonic_dynamics(t=dt, state=[q, p], m=m, k=k)
    q = q + dq_dt * dt
    p = p + dp_dt * dt
    init_state = [q, p]
    q_vals.append(q)
    p_vals.append(p)

# Plotting
t = list(range(200))
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# --- First Subplot (axs[0]) ---
axs[0].plot(p_vals, q_vals, color="blue", label="p v q")
axs[0].set_title("Momentum vs. Position, $p v. q$")
axs[0].set_xlabel("Momentum, $p$")
axs[0].set_ylabel("Position, $q$")
axs[0].legend(loc="best")  # Places the legend in the "best" non-overlapping location

# --- Second Subplot (axs[1]) ---
axs[1].plot(t, q_vals, color="red", linestyle="--", label="q v t")
axs[1].set_title("Position vs. Time, $q v. t$")
axs[1].set_xlabel("time, $t$")
axs[1].set_ylabel("Position, $q$")
axs[1].legend(loc="upper right")

# Optional: Add a main title to the entire figure
fig.suptitle("Plotting Hamiltonian Dynamics")

# Adjust layout to prevent labels/titles from overlapping
plt.tight_layout()  # Adjust rect to make space for subtitle

# Display the plot
plt.show()

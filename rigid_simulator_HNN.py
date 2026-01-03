# Here, we have an already trained HNN
# We use it to plot trajectory
import torch
from matplotlib import pyplot as plt

from HNN import HNN
from hnn_model_derivs import get_model_time_derivatives

device = "cuda" if torch.cuda.is_available() else "cpu"

# different initial states
init_a = torch.tensor([0.5, 0]).to(device)  # small release
init_b = torch.tensor([3.1, 0]).to(device)  # close to top
init_c = torch.tensor([0, 5.0]).to(device)  # kicked from near bottom

# params
m = 1.0  # mass
l = 1.0  # length of rod
dt = 0.05  # time-step
g = 9.8  # gravitational acceleration

hamiltonian_model = HNN().to(device)
hamiltonian_model.load_state_dict(torch.load("hamiltonian_nn_1.pth", weights_only=True))


def run_simulation_HNN(init_state: torch.Tensor):
    p_vals = []
    q_vals = []
    for i in range(1_000):
        curr_state_tensor = (
            init_state.clone().detach().unsqueeze(0).requires_grad_(True)
        )
        derivs = get_model_time_derivatives(hamiltonian_model, curr_state_tensor)

        dq_dt = derivs[0, 0]
        dp_dt = derivs[0, 1]

        q_old, p_old = init_state

        p_new = p_old + dp_dt * dt
        q_new = q_old + dq_dt * dt

        init_state = torch.tensor([q_new, p_new]).to(device)

        q_vals.append(q_new.item())
        p_vals.append(p_new.item())
    return p_vals, q_vals


a_p_vals, a_q_vals = run_simulation_HNN(init_a)
b_p_vals, b_q_vals = run_simulation_HNN(init_b)
c_p_vals, c_q_vals = run_simulation_HNN(init_c)

# Plotting
plt.plot(a_q_vals, a_p_vals, label="$q_0=0, p_0=0$")
plt.plot(b_q_vals, b_p_vals, label="$q_0=3.1, p_0=0$")
plt.plot(c_q_vals, c_p_vals, label="$q_0=0, p_0=5.0$")

plt.xlabel("$q$")
plt.ylabel("$p$")
plt.title("$p  v. q$ for different inital conditions simulated via Hamiltonian NN")
plt.legend()
plt.tight_layout()
plt.savefig("FIG5.png")

import torch

from pendulum_tensor import pendulum_dynamics_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

# params
m = 1.0  # mass
l = 1.0  # length of rod
dt = 0.05  # time-step
g = 9.8  # gravitational acceleration

mean = torch.tensor([-3.0, 3.0])
std = 0.1
init_states = (
    torch.normal(mean=mean.expand(1_000, 2), std=std).to(device).requires_grad_()
)

true_derivatives = pendulum_dynamics_tensor(t=dt, state=init_states, m=m, l=l, g=g)

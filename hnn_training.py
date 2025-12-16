import torch
from tqdm import tqdm

from HNN import HNN
from hnn_model_derivs import get_model_time_derivatives
from pendulum_tensor import pendulum_dynamics_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

# params
m = 1.0  # mass
l = 1.0  # length of rod
dt = 0.05  # time-step
g = 9.8  # gravitational acceleration

mean = torch.tensor([-3.0, 3.0])
std = 0.1
init_states = (6 * torch.rand(1_000, 2) - 3).to(device).requires_grad_()

true_derivatives = pendulum_dynamics_tensor(t=dt, state=init_states, m=m, l=l, g=g)

hamiltonian_nn = HNN().to(device)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(hamiltonian_nn.parameters(), lr=1e-2)

n_epochs = 1_200

for epoch in tqdm(range(n_epochs + 1)):
    deriv_pred = get_model_time_derivatives(hamiltonian_nn, init_states)
    loss = loss_func(deriv_pred, true_derivatives)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)

    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}\t Loss: {loss}")

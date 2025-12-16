import torch


def pendulum_dynamics_tensor(t, state: torch.Tensor, m, l, g) -> torch.Tensor:
    """
    inputs:
    t: The current time, solvers expect it.
    state: A list or array containing [q, p].
    m: mass
    l: the length of the rigid pendulum
    g: gravitational acceleration
    output:
    the list [dq_dt, dp_dt]
    """
    dq_dt = (state[:, 1] / (m * torch.pow(torch.tensor(l), 2))).unsqueeze(1)
    dp_dt = (-m * g * l * torch.sin(state[:, 0])).unsqueeze(1)
    return torch.cat([dq_dt, dp_dt], dim=1)

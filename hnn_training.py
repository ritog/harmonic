import torch

from HNN import HNN

device = "cuda" if torch.cuda.is_available() else "cpu"

mean = torch.tensor([0.0, 2.0])
std = 0.1
init_states = (
    torch.normal(mean=mean.expand(100, 2), std=std).to(device).requires_grad_()
)

hnn_model = HNN().to(device)


def get_model_time_derivatives(model, x):
    """
    Compute time derivatives [dq/dt, dp/dt] for a batch of inputs.
    x: Tensor of shape (Batch_Size, 2)
    """
    H_hat = model(x)

    # We sum() the energy to get a scalar, but because samples are independent,
    # the gradients separate out perfectly per row.
    grads = torch.autograd.grad(H_hat.sum(), x, create_graph=True)[0]

    # grads shape: (Batch, 2) -> [dH/dq, dH/dp]

    # flipping (Symplectic Swap (Hamilton's Eqs))
    # dq/dt =  dH/dp
    # dp/dt = -dH/dq

    dH_dq = grads[:, 0].unsqueeze(1)
    dH_dp = grads[:, 1].unsqueeze(1)

    return torch.cat([dH_dp, -dH_dq], dim=1)  # - because minus


deriv_pairs = get_model_time_derivatives(hnn_model, init_states)
print(deriv_pairs)

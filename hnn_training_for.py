import torch

from HNN import HNN

device = "cuda" if torch.cuda.is_available() else "cpu"

mean = torch.tensor([0.0, 2.0])
std = 0.1
init_states = (
    torch.normal(mean=mean.expand(100, 2), std=std).to(device).requires_grad_()
)

hnn_model = HNN().to(device)


def get_time_derivatives(model, x):
    grads_ls = []
    for qp_pair in x:
        h_hat = model(qp_pair)
        grads = torch.autograd.grad(h_hat, qp_pair, create_graph=True)
        grads_ls.append(grads)
    return grads_ls


h_hats = get_time_derivatives(hnn_model, init_states)
print(h_hats)


def get_deriv_pairs(h_hats):
    """
    Gets the pairs of [dq_dt, dp_dt]
    """
    deriv_pairs = []
    for h_hat in h_hats:
        dpdq = h_hat[0] * torch.tensor([-1.0, 1.0]).to("cuda")
        dqdp = torch.flip(dpdq, dims=[0])
        deriv_pairs.append(dqdp)
    return deriv_pairs


deriv_pairs = get_deriv_pairs(h_hats)
print(deriv_pairs)

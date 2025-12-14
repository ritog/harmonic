from typing import List


def harmonic_dynamics(t, state: List, m, k) -> List:
    """
    inputs:
    t: The current time, solvers expect it.
    state: A list or array containing [q, p].
    m, k: The mass and spring constant.
    output:
    the list [dq_dt, dp_dt]
    """
    dq_dt = state[1] / m
    dp_dt = -k * state[0]
    return [dq_dt, dp_dt]

from typing import List

import numpy as np


def pendulum_dynamics(t, state: List, m, l, g) -> List:
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
    dq_dt = state[1] / (m * np.power(l, 2))
    dp_dt = -m * g * l * np.sin(state[0])
    return [dq_dt, dp_dt]

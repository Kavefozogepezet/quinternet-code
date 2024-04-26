from qubit import *
import numpy as np


def apply_tvapd(state, target, dt, t1, t2=None):
    '''
    Applies time-varying amplitude damping to a qubit.

    Parameters:
    ---
    - state: The quantum state on which the operation is applied,
    - target: The index of the target qubit,
    - dt: The time spent in the tvad channel,
    - t1: The relaxation time constant,
    - t2: The dephasing time constant (optional, default is t1/2).

    '''
    g = 1 - np.e**(-dt/t1)
    l = 0 if t2 is None else 1 - np.e**(dt/t1 - 2*dt/t2)

    E0 = Gate([
        [1, 0],
        [0, np.sqrt((1-g)*(1-l))]
    ])
    E1 = Gate([
        [0, np.sqrt(g)],
        [0, 0]
    ])
    E2 = Gate([
        [0, 0],
        [0, np.sqrt((1-g)*l)]
    ])

    apply_kraus(state, target, E0, E1, E2)


def apply_depol(state, target, L, alpha):
    p = 1 - 10**(-alpha*L/10)

    E0 = Gate(np.eye(2) * np.sqrt(1-p))
    E1 = Gate(Gate.X.mat * np.sqrt(p/3))
    E2 = Gate(Gate.Y.mat * np.sqrt(p/3))
    E3 = Gate(Gate.Z.mat * np.sqrt(p/3))

    apply_kraus(state, target, E0, E1, E2, E3)

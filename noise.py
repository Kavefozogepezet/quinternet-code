from qubit import *
from gate import *
import numpy as np


def apply_tvapd(state, target, dt, t1, t2=None):
    '''
    Applies time-varying amplitude damping to a qubit.

    Parameters
    ----------
    state: QState
        The quantum state on which the operation is applied,
    target: int
        The index of the target qubit,
    dt: float
        The time spent in the tvad channel,
    t1: float
        The relaxation time,
    t2: float
        The dephasing time (optional, default is 2*t1).
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

    apply_kraus_to(state, target, E0, E1, E2)


def apply_depol(state, target, L, alpha):
    '''
    Applies depolarizing noise to a qubit.

    Parameters
    ----------
    state: QState
        The quantum state on which the operation is applied,
    target: int
        The index of the target qubit,
    L: float
        The length of the medium,
    alpha: float
        The attenuation of the medium.
    '''
    p = 1 - 10**(-alpha*L/10)

    E0 = Gate(np.eye(2) * np.sqrt(1-p))
    E1 = Gate(Gate.X.mat * np.sqrt(p/3))
    E2 = Gate(Gate.Y.mat * np.sqrt(p/3))
    E3 = Gate(Gate.Z.mat * np.sqrt(p/3))

    apply_kraus_to(state, target, E0, E1, E2, E3)

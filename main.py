from qubit import *
from noise import *
from gate import *
import numpy as np
from matplotlib import pyplot as plt
import qutip as q


if __name__ == '__main__':
    bloch = q.Bloch()

    for i in range(20):
        state = QState(qubits=1)
        apply_unitary(state, Gate.H)

        #apply_depol(state, 1, i, 0.2)
        apply_tvapd(state, 0, i/500, 0.01, 0.02)

        u = 2 * state.mat[0, 1].real
        v = 2 * state.mat[1, 0].imag
        w = state.mat[0, 0].real - state.mat[1, 1].real

        bloch.add_points([u, v, w])
        print(np.trace(state.mat))

    bloch.show()
    #plt.plot(angles, cs)
    plt.show()
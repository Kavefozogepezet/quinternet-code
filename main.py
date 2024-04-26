from qubit import *
from noise import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    cs = []
    angles = []

    print(Gate.X.c(up=False))

    for i in range(100):
        state = QState(qubits=4)
        apply_unitary(state, [Gate.H, 3])
        apply_unitary(state, [Gate.X.c(), 2])

        apply_unitary(state, [3, Gate.H])
        apply_unitary(state, [2, Gate.X.c(up=False)])

        print(state)

        #apply_depol(state, 1, i, 0.2)
        apply_tvapd(state, 1, i/10000, 0.01)
        print(np.trace(state.state))
        cs.append(state.concurrence())
        angles.append(i)

    #plt.plot(angles, cs)
    #plt.show()
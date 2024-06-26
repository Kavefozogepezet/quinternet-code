from qubit import *
from gate import *
from noise import *


def trace_out_middle(state):
    result = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        I = np.eye(2)
        base = np.zeros((1, 4))
        base[0,i] = 1
        mat1 = np.kron(np.kron(I, base), I)
        mat2 = np.kron(np.kron(I, base.T), I)
        result += mat1 @ state.mat @ mat2
    state.mat = result


def projection_to(base):
    proj = QState.get_base(2, base).mat
    I = np.eye(2)
    return np.kron(np.kron(I, proj), I)


def swap_entanglement(state):
    apply_unitary(state, [1, Gate.X.c(), 1])
    apply_unitary(state, [1, Gate.H, 2])

    Z1 = compose_unitary(Gate.Z, 3).mat
    X4 = compose_unitary(3, Gate.X).mat
    Z1X4 = compose_unitary(Gate.Z, 2, Gate.X).mat

    K0 = Gate(projection_to(0b00))
    K1 = Gate(X4 @ projection_to(0b01))
    K2 = Gate(Z1 @ projection_to(0b10))
    K3 = Gate(Z1X4 @ projection_to(0b11))

    apply_kraus(state, K0, K1, K2, K3)


def sim_single_connection(length, refrac, alpha, t1, t2, sim_atennuation=False, sim_noise=True):
    state = QState(qubits=4)
    apply_unitary(state, [Gate.H, 2, Gate.H])
    apply_unitary(state, [Gate.X.c(), Gate.X.c(up=False)])

    if sim_noise:
        dt = length * refrac / 300_000
        apply_tvapd(state, 0, dt, t1, t2)
        apply_depol(state, 1, length, alpha)
        apply_depol(state, 2, length, alpha)
        apply_tvapd(state, 0, dt, t1, t2)

    swap_entanglement(state)
    trace_out_middle(state)

    if not sim_atennuation:
        return state
    else:
        tries = 0
        p_loss = 1 - 10**(-alpha*length/10)
        while tries < 100:
            tries += 1
            p1, p2 = np.random.rand(2)
            if p1 > p_loss and p2 > p_loss:
                return state, tries


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import qutip as q
    import plotutils as pu

    state = sim_single_connection(10, 1.5, 0.1, 0.001, 0.002)
    fig, ax = q.matrix_histogram(state.mat, bar_style='abs', colorbar=False)
    ax.xaxis.set_tick_params(labelsize=10, pad=-5)
    ax.yaxis.set_tick_params(labelsize=10, pad=-5)
    pu.size_correction(fig, sy=0.05)
    pu.export_plot(fig, 'direct_mat.pgf', 3)
    plt.clf()

    lengths = np.linspace(0, 20, 11)
    fy = np.zeros_like(lengths)
    cy = np.zeros_like(lengths)

    ideal = np.zeros((4, 1))
    ideal[0] = ideal[3] = 1/np.sqrt(2)
    
    for i, length in enumerate(lengths):
        state = sim_single_connection(length, 1.5, 0.1, 0.001, 0.002)
        fy[i] = state.fidelity(ideal)
        cy[i] = state.concurrence()

    plt.plot(lengths, fy, label='fidelity', marker='s')
    plt.plot(lengths, cy, label='concurrence', marker='^')
    plt.xlabel('hossz (km)')
    plt.ylabel('fidelity / concurrence')
    plt.ylim(0, 1.2)
    legend = plt.legend()
    pu.styled_legend(legend)
    pu.export_plot(fig, 'direct_fc.pgf', 3)

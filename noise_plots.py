from qubit import *
from noise import *
from gate import *
import numpy as np
from matplotlib import pyplot as plt
import qutip as q
import plotutils as pu


def bloch_style(bloch):
    bloch.frame_alpha = 0
    bloch.sphere_alpha = 0.075
    bloch.font_size = 11
    bloch.xlabel = ['$\\left|+\\right>$', '']
    bloch.ylabel = ['$\\left|i\\right>$', '']


def apd_plot():
    bloch = q.Bloch(view=[-20, 20])
    colors = ['b', 'r', 'g', 'orange']
    times = np.logspace(-1.4, -3, 7)[::-1]
    times = np.concatenate(([0], times))
    ratios = [0.2, 0.5, 1, 2]

    for ratio, color in zip([0.2, 0.5, 1, 1.5, 2], colors):
        points = np.zeros((3, len(times)))
        for i, dt in enumerate(times):
            state = QState(qubits=1)
            apply_unitary(state, Gate.H)
            apply_tvapd(state, 0, dt, 0.01, 0.01 * ratio)
            points[:,i] = state.bloch()
    
        bloch.add_points(points, colors=color)
        for i in range(points.shape[1] - 1):
            bloch.add_line(
                points[:,i], points[:,i+1],
                color=color)
    
    bloch_style(bloch)
    bloch.show()
    legend_handles = [
        plt.Line2D([], [], color=c, marker=m, markerfacecolor=c, label=l)
        for c, l, m in zip(colors, [f'{r}' for r in ratios], ['o', 's', 'd', '^'])
    ]
    legend = bloch.fig.legend(handles=legend_handles, loc='upper right')
    pu.size_correction(bloch.fig, sy=0.05)
    pu.styled_legend(legend)
    pu.export_plot(bloch.fig, 'tvapd.pgf', 2.5)
    #plt.show()


def depol_plot():
    bloch = q.Bloch(view=[-20, 20])
    lengths = np.linspace(0, 50, 8)
    colors = ['b', 'r', 'g']

    qubits = (QState(qubits=1), QState(qubits=1), QState(qubits=1))
    apply_unitary(qubits[0], Gate.H)
    apply_unitary(qubits[1], Gate.H)
    apply_unitary(qubits[2], Gate.H)
    apply_unitary(qubits[1], Gate.Ry(np.pi/6))
    apply_unitary(qubits[2], Gate.Ry(-np.pi/6))

    for qubit, color in zip(qubits, colors):
        points = np.zeros((3, len(lengths)))
        for i, L in enumerate(lengths):
            state = QState(init_mat=qubit.mat)
            apply_depol(state, 0, L, 0.2)
            points[:,i] = state.bloch()
    
        bloch.add_points(points)
        for i in range(points.shape[1] - 1):
            bloch.add_line(points[:,i], points[:,i+1], color=color)
    
    bloch_style(bloch)
    bloch.show()
    labels = ['$\\left|+\\right>$',
              '$\\left|\\alpha\\right>$',
              '$\\left|\\beta\\right>$']
    legend_handles = [
        plt.Line2D([], [], color=c, marker=m, markerfacecolor=c, label=l)
        for c, l, m in zip(colors, labels, ['o', 's', 'd'])
    ]
    legend = bloch.fig.legend(handles=legend_handles, loc='upper right')
    pu.size_correction(bloch.fig, sy=0.05)
    pu.styled_legend(legend)
    pu.export_plot(bloch.fig, 'depol.pgf', 2.5)
    #plt.show()


if __name__ == '__main__':
    apd_plot()
    depol_plot()

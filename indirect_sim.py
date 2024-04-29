from qubit import *
from gate import *
from noise import *
from direct_sim import *


def sim_multiple_coinnection(chain_length, fibre_length, refrac, alpha, t1, t2, sim_atennuation=True, sim_noise=True):
    result = sim_single_connection(fibre_length, refrac, alpha, t1, t2, sim_atennuation, sim_noise)
    tries = 0
    if sim_atennuation:
        result, tries = result

    dt = 2*fibre_length*refrac/300_000
    full_t = 2 * dt * (chain_length-2)

    if sim_noise:
        apply_tvapd(result, 0, full_t, t1, t2)

    for i in range(chain_length-2):
        state = sim_single_connection(fibre_length, refrac, alpha, t1, t2, sim_atennuation, sim_noise)
        if sim_atennuation and sim_noise:
            state, t = state
            if tries < t:
                apply_tvapd(result, 0, dt*(t-tries), t1, t2)
                apply_tvapd(result, 1, dt*(t-tries), t1, t2)
                tries = t
            elif tries > t:
                apply_tvapd(state, 0, dt*(tries-t), t1, t2)
                apply_tvapd(state, 1, dt*(tries-t), t1, t2)
        result = QState.combine(result, state)
        if sim_noise and i == chain_length-3:
            apply_tvapd(result, 3, full_t, t1, t2)
        swap_entanglement(result)
        trace_out_middle(result)

    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import plotutils as pu

    distance = 10
    shots = 1000
    device_counts = np.arange(2, 11, 1)
    fy = np.zeros((len(device_counts), shots))
    cy = np.zeros((len(device_counts), shots))

    ideal = np.zeros((4, 1))
    ideal[0] = ideal[3] = 1/np.sqrt(2)

    for i, d in enumerate(device_counts):
        cable_len = distance/(d-1)/2
        print(f'progress: {i}/{len(device_counts)}, cable length: {cable_len} km')
        for shot in range(shots):
            state = sim_multiple_coinnection(d, cable_len, 1.5, 0.1, 0.001, 0.002, sim_atennuation=True)
            fy[i, shot] = state.fidelity(ideal)
            cy[i, shot] = state.concurrence()

    fig, ax = plt.subplots()

    row_means = np.mean(fy, axis=1)
    row_errm = np.maximum(0, row_means - np.min(fy, axis=1))
    row_errp = np.maximum(0, np.max(fy, axis=1) - row_means)

    ax.errorbar(device_counts, row_means, yerr=[row_errm, row_errp], label='fidelity', fmt='s', capsize=5, markersize=3)
    ax.axhline(row_means[0], linestyle='--', linewidth=1)

    row_means = np.mean(cy, axis=1)
    row_errm = np.maximum(0, row_means - np.min(cy, axis=1))
    row_errp = np.maximum(0, np.max(cy, axis=1) - row_means)

    ax.errorbar(device_counts, row_means, yerr=[row_errm, row_errp], label='concurrence', fmt='s', capsize=5, markersize=3)
    ax.axhline(row_means[0], color='orange', linestyle='--', linewidth=1)

    ax.grid(axis='x')
    ax.set_xlabel('Eszköz lánc hossza')
    ax.set_ylabel('fidelity / concurrence')
    ax.set_xticks(range(2, 11))
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(top=1)
    legend = ax.legend()
    pu.styled_legend(legend)
    pu.export_plot(fig, 'indirect.pgf', 3)
    #plt.show()


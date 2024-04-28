import numpy as np
from tabulate import tabulate


class QState:
    '''
    Represents the state of an n-qubit quantum system as a density matrix.
    '''
    def __init__(self, **kwargs):
        if 'init_mat' in kwargs:
            self.mat = kwargs['init_mat']
        else:
            if 'init_state' in kwargs:
                init_state = kwargs['init_state']
            elif 'qubits' in kwargs:
                init_state = np.zeros(2**kwargs['qubits'])
                init_state[0] = 1
            self.mat = np.tensordot(init_state, init_state, 0)


    def __str__(self):
        bits = int(np.log2(self.mat.shape[0]))
        names = [f'|{i:0{bits}b}>' for i in range(self.mat.shape[0])]
        probs = []
        for i in range(self.mat.shape[0]):
            probs.append(f'{self.mat[i, i]:.2f}')
        return tabulate(zip(names, probs), headers=['State', 'Probability'], tablefmt='pretty')
    

    def concurrence(self):
        from gate import Gate
        if self.mat.shape[0] != 4:
            raise ValueError('Concurrence is only defined for two-qubit states.')
        
        rho_star = np.conj(self.mat)
        sigma_y = np.kron(Gate.Y.mat, Gate.Y.mat)
        R = self.mat @ sigma_y @ rho_star @ sigma_y
        eigvals = np.sqrt(np.linalg.eigvals(R))
        eigvals = np.sort(eigvals, 0)[::-1]
        return max(0, eigvals[0] - np.sum(eigvals[1:]))
    

    def fidelity(self, ideal):
        return (ideal.conj().T @ self.mat @ ideal).real
    

    def bloch(self):
        u = 2 * self.mat[0, 1].real
        v = 2 * self.mat[1, 0].imag
        w = self.mat[0, 0].real - self.mat[1, 1].real
        return u, v, w
    

    def get_base(qubits, base):
        mat = np.zeros((2**qubits, 2**qubits))
        mat[base, base] = 1
        return QState(init_mat=mat)

import numpy as np
from tabulate import tabulate


class QState:
    '''
    Behind the scenes class to represent the state of the system
    using a density matrix.

    This class should not be used directly.
    '''
    def __init__(self, **kwargs):
        if 'init_state' in kwargs:
            init_state = kwargs['init_state']
        elif 'qubits' in kwargs:
            init_state = np.zeros(2**kwargs['qubits'])
            init_state[0] = 1
        self.state = np.tensordot(init_state, init_state, 0)


    def __str__(self):
        bits = int(np.log2(self.state.shape[0]))
        names = [f'|{i:0{bits}b}>' for i in range(self.state.shape[0])]
        probs = []
        for i in range(self.state.shape[0]):
            probs.append(f'{self.state[i, i]:.2f}')
        return tabulate(zip(names, probs), headers=['State', 'Probability'], tablefmt='pretty')
    

    def concurrence(self):
        if self.state.shape[0] != 4:
            raise ValueError('Concurrence is only defined for two-qubit states.')
        
        rho_star = np.conj(self.state)
        sigma_y = np.kron(Gate.Y.mat, Gate.Y.mat)
        R = self.state @ sigma_y @ rho_star @ sigma_y
        eigvals = np.sqrt(np.linalg.eigvals(R))
        eigvals = np.sort(eigvals, 0)[::-1]
        return max(0, eigvals[0] - np.sum(eigvals[1:]))
    

# TODO proper implementation of gates
class Gate:
    def __init__(self, matrix):
        self.mat = np.matrix(matrix)


    def Ry(theta):
        return Gate([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])


    def c(self, up=True):
        result = np.eye(self.mat.shape[0] + 2)
        result[-2:,-2:] = self.mat
        if not up:
            result = Gate.SWAP.mat @ result @ Gate.SWAP.mat
        return Gate(result)


    def adjoint(self):
        return Gate(self.mat.conj().T)
    

    def __str__(self) -> str:
        mat_str = []
        cols = [0] * self.mat.shape[1]
        for i in range(self.mat.shape[0]):
            mat_str.append([])
            for j in range(self.mat.shape[1]):
                mat_str[i].append(str(round(self.mat[i,j], 4)))
                cols[j] = max(cols[j], len(mat_str[i][-1]))
        
        mat = '┌' + ' ' * (sum(cols) + len(cols) * 2) + '┐'
        for i in range(self.mat.shape[0]):
            mat += '\n│ '
            for j in range(self.mat.shape[1]):
                mat += mat_str[i][j].ljust(cols[j] + 2)
            mat = mat[:-1] + '│'
        mat += '\n└' + ' ' * (sum(cols) + len(cols) * 2) + '┘'
        return mat


Gate.X = Gate([[0, 1], [1, 0]])
Gate.Y = Gate([[0, -1j], [1j, 0]])
Gate.Z = Gate([[1, 0], [0, -1]])
Gate.H = Gate(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
Gate.SWAP = Gate([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    

def compose_unitary(*unitaries):
    result = np.eye(2**unitaries[0]) if isinstance(unitaries[0], int) else unitaries[0].mat
    for unitary in unitaries[1:]:
        if isinstance(unitary, int):
            result = np.kron(result, np.eye(2**unitary))
        else:
            result = np.kron(result, unitary.mat)
    return Gate(result)


def combine_unitaries(*unitaries):
    result = unitaries[0].mat
    for unitary in unitaries[1:]:
        result = np.dot(result, unitary.mat)
    return Gate(result)


def apply_unitary(qstate, unitary: Gate|list):
    if isinstance(unitary, list):
        unitary = compose_unitary(*unitary)
    qstate.state = np.dot(np.dot(unitary.mat, qstate.state), unitary.adjoint().mat)


def apply_kraus(qstate, target: int, *kraus):
    result = np.zeros(qstate.state.shape, dtype=complex)
    pre = np.eye(2**target)
    post = np.eye(qstate.state.shape[0]//2**(target + 1))
    for k in kraus:
        big_k = np.kron(np.kron(pre, k.mat), post)
        result += np.dot(np.dot(big_k, qstate.state), big_k.conj().T)
    qstate.state = result
            

from qubit import QState
import numpy as np


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
    

def compose_unitary(*unitaries: Gate|int):
    result = np.eye(2**unitaries[0]) if isinstance(unitaries[0], int) else unitaries[0].mat
    for unitary in unitaries[1:]:
        if isinstance(unitary, int):
            result = np.kron(result, np.eye(2**unitary))
        else:
            result = np.kron(result, unitary.mat)
    return Gate(result)


def combine_unitaries(*unitaries: Gate|list[Gate|int]):
    result = unitaries[0] if isinstance(unitaries[0], Gate) else compose_unitary(*unitaries[0])
    for unitary in unitaries[1:]:
        if isinstance(unitary, list):
            unitary = compose_unitary(*unitary)
        result = np.dot(result, unitary.mat)
    return Gate(result)


def apply_unitary(qstate: QState, unitary: Gate|list[Gate|int]):
    if isinstance(unitary, list):
        unitary = compose_unitary(*unitary)
    qstate.mat = np.dot(np.dot(unitary.mat, qstate.mat), unitary.adjoint().mat)


def apply_kraus(state: QState, *kraus: Gate):
    result = np.zeros(state.mat.shape, dtype=complex)
    for k in kraus:
        result += np.dot(np.dot(k.mat, state.mat), k.adjoint().mat)
    state.mat = result


def apply_kraus_to(qstate: QState, target: int, *kraus: Gate):
    pre = np.eye(2**target)
    post = np.eye(qstate.mat.shape[0]//2**(target + 1))
    big_kraus = [ Gate(np.kron(np.kron(pre, k.mat), post)) for k in kraus]
    apply_kraus(qstate, *big_kraus)
            
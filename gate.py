from qubit import QState
import numpy as np


class Gate:
    '''
    A class representing a quantum gate.

    Attributes
    ----------
    mat: np.ndarray
        The matrix representation of the gate, a 2^n x 2^n matrix, where n is the number of qubits the gate acts on.
    
    Methods
    -------
    c(up: bool=True):
        Returns the controlled version of the gate.
        If up is True, the control qubit is the first qubit, otherwise it is the last qubit.
    adjoint():
        Returns the adjoint of the gate.
    
    Static Methods
    --------------
    Ry(theta: float) / Rx(theta: float) / Rz(theta: float):
        Returns the rotating gate with angle theta around the corresponding axis (R<axis>).

    Static Attributes
    -----------------
    X / Y / Z / H / SWAP: Gate
        The Pauli-X, Pauli-Y, Pauli-Z, Hadamard, and SWAP gates.
    '''
    def __init__(self, matrix):
        self.mat = np.matrix(matrix)


    def Ry(theta):
        '''
        Returns the rotating gate with angle theta around the y-axis.

        Parameters
        ----------
        theta: float
            The angle of rotation in radians.

        Returns
        -------
        Gate: The rotating gate.
        '''
        return Gate([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
    
    def Rx(theta):
        '''
        Returns the rotating gate with angle theta around the x-axis.

        Parameters
        ----------
        theta: float
            The angle of rotation in radians.

        Returns
        -------
        Gate: The rotating gate.
        '''
        return Gate([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ])
    
    def Rz(theta):
        '''
        Returns the rotating gate with angle theta around the z-axis.
        
        Parameters
        ----------
        theta: float
            The angle of rotation in radians.
            
        Returns
        -------
        Gate: The rotating gate.
        '''
        return Gate([
            [np.e**(-1j*theta/2), 0],
            [0, np.e**(1j*theta/2)]
        ])


    def c(self, up=True):
        '''
        Calculates the controlled version of the gate.
        
        Parameters
        ----------
        up: bool
            If True, the control qubit is the first qubit, otherwise it is the last qubit.
            
        Returns
        -------
        Gate: The controlled gate.
    '''
        result = np.eye(self.mat.shape[0] + 2)
        result[-2:,-2:] = self.mat
        if not up:
            result = Gate.SWAP.mat @ result @ Gate.SWAP.mat
        return Gate(result)


    def adjoint(self):
        '''Returns the adjoint of the gate.'''
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
    '''
    Composes a unitary gate from a list of unitaries that acts on a quantum register.
    The order of the gates in the list determines the index of the qubits they act on.

    Parameters
    ----------
    *unitaries: list[Gate|int]
        A list of unitary gates or the number of qubits they act on.
        If an integer n, the gate is the identity gate acting on n qubits.

    Returns
    -------
    Gate: The thensor products of the unitaries in order.
    '''
    result = np.eye(2**unitaries[0]) if isinstance(unitaries[0], int) else unitaries[0].mat
    for unitary in unitaries[1:]:
        if isinstance(unitary, int):
            result = np.kron(result, np.eye(2**unitary))
        else:
            result = np.kron(result, unitary.mat)
    return Gate(result)


def combine_unitaries(*unitaries: Gate|list[Gate|int]):
    '''
    Combines a list of unitaries as they were acting on a register in the order they appeare in the list.

    Parameters
    ----------
    *unitaries: list[Gate|list[Gate|int]]:
        A list of unitaries or lists of unitaries that act on a quantum register.
        If a list, the unitaries are composed into a single unitary (see compose_unitary).

    Returns
    -------
    Gate: The product of the unitaries in the order they appear in the list.
    '''
    result = unitaries[0] if isinstance(unitaries[0], Gate) else compose_unitary(*unitaries[0])
    for unitary in unitaries[1:]:
        if isinstance(unitary, list):
            unitary = compose_unitary(*unitary)
        result = np.dot(result, unitary.mat)
    return Gate(result)


def apply_unitary(qstate: QState, unitary: Gate|list[Gate|int]):
    '''
    Applies a unitary gate to a quantum state.

    Parameters
    ----------
    qstate: QState
        The quantum state on which the operation is applied.
    unitary: Gate|list[Gate|int]
        The unitary gate or the list of unitaries acting on the state.
        If a list, the unitaries are composed into a single unitary (see compose_unitary).
    '''
    if isinstance(unitary, list):
        unitary = compose_unitary(*unitary)
    qstate.mat = np.dot(np.dot(unitary.mat, qstate.mat), unitary.adjoint().mat)


def apply_kraus(state: QState, *kraus: Gate):
    '''
    Applies a superoperator given by a list of Kraus operators to a quantum state.
    
    Parameters
    ----------
    state: QState
        The quantum state on which the operation is applied.
    *kraus: list[Gate]
        The list of the Kraus operators.
    '''
    result = np.zeros(state.mat.shape, dtype=complex)
    for k in kraus:
        result += np.dot(np.dot(k.mat, state.mat), k.adjoint().mat)
    state.mat = result


def apply_kraus_to(qstate: QState, target: int, *kraus: Gate):
    '''
    Applies a superoperator given by a list of Kraus operators to a single qubit in a quantum state.
    
    Parameters
    ----------
    qstate: QState
        The quantum state that holds the target qubit.
    target: int
        The index of the qubit on which the operation is applied.
    *kraus: list[Gate]
        The list of the Kraus operators.
    '''
    pre = np.eye(2**target)
    post = np.eye(qstate.mat.shape[0]//2**(target + 1))
    big_kraus = [ Gate(np.kron(np.kron(pre, k.mat), post)) for k in kraus]
    apply_kraus(qstate, *big_kraus)
            
import numpy as np
from tabulate import tabulate


class QState:
    '''
    Represents the state of an n-qubit quantum system as a density matrix.

    Attributes
    ----------
    mat: np.ndarray
        The density matrix of the quantum state. It is a 2^n x 2^n matrix, where n is the number of qubits.

    Methods
    -------
    concurrence():
        Returns the concurrence of a two-qubit quantum state.
        Throws a ValueError if the state is not two-qubit.
    fidelity(ideal: np.ndarray):
        Returns the fidelity of the quantum state compared to an ideal state.
    bloch():
        Returns the Bloch vector of a single-qubit quantum state.
        Throws a ValueError if the state is not single-qubit.

    Static Methods
    --------------
    get_base(qubits: int, base: int):
        Returns a quantum state in the specified base state.
    combine(*states):
        Returns the tensor product of multiple quantum states.
    '''
    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        init_mat (optional): np.ndarray
            The initial density matrix of the quantum state, a 2^n x 2^n matrix.
        init_state (optional): np.ndarray
            The initial state vector of the quantum state, a 2^n x 1 matrix.
        qubits (optional): int
            The number of qubits in the quantum state, initialized to the zero state.

        **note:** Only one of the three parameters should be provided.

        Raises
        ------
        TypeError: If more than one, or incorrectly named parameters are provided.
        '''
        if len(kwargs) != 1:
            raise TypeError('QState constructor takes exactly one keyword argument.')

        if 'init_mat' in kwargs:
            self.mat = kwargs['init_mat']
        else:
            if 'init_state' in kwargs:
                init_state = kwargs['init_state']
            elif 'qubits' in kwargs:
                init_state = np.zeros(2**kwargs['qubits'])
                init_state[0] = 1
            else:
                for key in kwargs.keys():
                    raise TypeError(f'Unknown paramether: {key}.')
            self.mat = np.tensordot(init_state, init_state, 0)


    def __str__(self):
        '''Returns a table with the basis states and their probabilities.'''
        bits = int(np.log2(self.mat.shape[0]))
        names = [f'|{i:0{bits}b}>' for i in range(self.mat.shape[0])]
        probs = []
        for i in range(self.mat.shape[0]):
            probs.append(f'{self.mat[i, i]:.2f}')
        return tabulate(zip(names, probs), headers=['State', 'Probability'], tablefmt='pretty')
    

    def concurrence(self) -> float:
        '''
        Returns the concurrence of a two-qubit quantum state.
        
        Raises
        ------
        ValueError: if the state is not two-qubit.
        '''
        from gate import Gate
        if self.mat.shape[0] != 4:
            raise ValueError('Concurrence is only defined for two-qubit states.')
        
        rho_star = np.conj(self.mat)
        sigma_y = np.kron(Gate.Y.mat, Gate.Y.mat)
        R = self.mat @ sigma_y @ rho_star @ sigma_y
        eigvals = np.sqrt(np.linalg.eigvals(R))
        eigvals = np.sort(eigvals, 0)[::-1]
        return max(0, eigvals[0] - np.sum(eigvals[1:]))
    

    def fidelity(self, ideal: np.ndarray) -> float:
        '''
        Calculates the fidelity of the quantum state compared to an ideal state.
        
        Parameters
        ----------
        ideal: np.ndarray
            The ideal state vector to compare the quantum state to, a 2^n x 1 matrix.

        Returns
        -------
        float: The fidelity between the quantum state and the ideal state.
        '''
        return float((ideal.conj().T @ self.mat @ ideal).real)
    

    def bloch(self) -> tuple[float, float, float]:
        '''
        Returns the Bloch vector of a single-qubit quantum state.

        Raises
        ------
        ValueError: if the state is not single-qubit.

        Returns
        -------
        tuple[float, float, float]: The Bloch vector of the quantum state.
        '''
        if self.mat.shape[0] != 2:
            raise ValueError('Bloch vector is only defined for single-qubit states.')
        u = 2 * self.mat[0, 1].real
        v = 2 * self.mat[1, 0].imag
        w = self.mat[0, 0].real - self.mat[1, 1].real
        return u, v, w
    

    def get_base(qubits: int, base: int):
        '''
        Returns a quantum state in the specified base state.
        
        Parameters
        ----------
        qubits: int
            The number of qubits in the quantum state.
        base: int
            In binary it reads the base state in reverse, e.g. 0b0100 for |0010>.

        Returns
        -------
        QState: The quantum state in the specified base state.
        '''
        mat = np.zeros((2**qubits, 2**qubits))
        mat[base, base] = 1
        return QState(init_mat=mat)
    

    def combine(*states):
        '''
        Calculates the tensor product of multiple quantum states.

        Parameters
        ----------
        *states: QState
            The quantum states to combine.
        
        Returns
        -------
        QState: The combined state.
        '''
        result = states[0]
        for state in states[1:]:
            result.mat = np.kron(result.mat, state.mat)
        return result

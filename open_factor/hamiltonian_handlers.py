from abc import abstractmethod

from qiskit.quantum_info import SparsePauliOp


class HamiltonianHandler:
    @abstractmethod
    def get_hamiltonian(self, n) -> SparsePauliOp:
        raise NotImplementedError()

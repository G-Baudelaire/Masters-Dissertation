from qiskit.quantum_info import SparsePauliOp


class HamiltonianHandler:
    def get_hamiltonian(self, n) -> SparsePauliOp:
        raise NotImplementedError()

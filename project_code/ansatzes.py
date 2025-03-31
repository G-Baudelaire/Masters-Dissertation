from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class AnsatzImplementation:
    def get_ansatz(self, p_bits: int, q_bits: int) -> QuantumCircuit:
        raise NotImplementedError()


class LinearCNotAnsatz(AnsatzImplementation):
    def get_ansatz(self, p_bits, q_bits):
        number_of_qubits = p_bits + q_bits
        quantum_circuit = QuantumCircuit(number_of_qubits)
        parameters = ParameterVector("θ", number_of_qubits * 3)

        self._first_layer(number_of_qubits, quantum_circuit, parameters)
        self._second_layer(number_of_qubits, quantum_circuit, parameters)
        self._third_layer(number_of_qubits, quantum_circuit, parameters)

        # Do not measure here; add measurements only in simulation
        return quantum_circuit

    def _first_layer(self, number_of_qubits, quantum_circuit, parameters):
        for i in range(number_of_qubits):
            quantum_circuit.h(i)
            quantum_circuit.ry(parameters[i], i)

        # CNOTs
        for i in range(number_of_qubits - 1):
            quantum_circuit.cx(i, i + 1)
        quantum_circuit.barrier()

    def _second_layer(self, number_of_qubits, quantum_circuit, parameters):
        for i in range(number_of_qubits):
            quantum_circuit.ry(parameters[(1 * number_of_qubits) + i], i)

        # CNOTs
        for i in range(number_of_qubits - 1):
            quantum_circuit.cx(i, i + 1)
        quantum_circuit.barrier()

    def _third_layer(self, number_of_qubits, quantum_circuit, parameters):
        for i in range(number_of_qubits):
            quantum_circuit.ry(parameters[(2 * number_of_qubits) + i], i)


class OptimisedLinearCNotAnsatz(LinearCNotAnsatz):
    def get_ansatz(self, p_bits, q_bits):
        return super().get_ansatz(p_bits - 1, q_bits - 1)

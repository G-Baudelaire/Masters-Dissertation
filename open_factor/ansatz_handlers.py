from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from open_factor.utils import Utils


class AnsatzHandler:
    def get_ansatz(self, n) -> QuantumCircuit:
        raise NotImplementedError()


class LinearCNOTAnsatzHandler(AnsatzHandler):
    def get_ansatz(self, n):
        p_bits, q_bits = Utils.bits_needed(n)
        number_of_qubits = p_bits + q_bits - 2
        quantum_circuit = QuantumCircuit(number_of_qubits)
        parameters = ParameterVector("θ", number_of_qubits * 3)

        self._first_layer(number_of_qubits, quantum_circuit, parameters)
        self._second_layer(number_of_qubits, quantum_circuit, parameters)
        self._third_layer(number_of_qubits, quantum_circuit, parameters)

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

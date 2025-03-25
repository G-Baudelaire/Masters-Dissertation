from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def get_ansatz(number_of_qubits):
    params = ParameterVector("θ", number_of_qubits)  # Parameterized angles
    qc = QuantumCircuit(number_of_qubits)

    # Apply RY rotations
    for i in range(number_of_qubits):
        qc.ry(params[i], i)

    # Apply linear entangling CNOTs
    for i in range(number_of_qubits - 1):
        qc.cx(i, i + 1)

    return qc, params

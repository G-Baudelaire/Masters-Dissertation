from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter


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

def custom_linear_cnot(number_of_qubits):
    quantum_circuit = QuantumCircuit(number_of_qubits)
    parameters = ParameterVector("θ", number_of_qubits * 3)

    # First layer: H + RY
    for i in range(number_of_qubits):
        quantum_circuit.h(i)
        quantum_circuit.ry(parameters[i], i)

    # CNOTs
    for i in range(number_of_qubits-1):
        quantum_circuit.cx(i, i + 1)
    quantum_circuit.barrier()

    # Second RY layer
    for i in range(number_of_qubits):
        quantum_circuit.ry(parameters[(1* number_of_qubits)+ i], i)

    # CNOTs
    for i in range(number_of_qubits-1):
        quantum_circuit.cx(i, i + 1)
    quantum_circuit.barrier()

    # Third RY layer
    for i in range(number_of_qubits):
        quantum_circuit.ry(parameters[(2* number_of_qubits)+ i], i)

    # Do not measure here; add measurements only in simulation
    return quantum_circuit, parameters
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

EXAMPLE_HAMILTONIAN = SparsePauliOp.from_list(
    [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
)

def get_ansatz():
    quantum_circuit = QuantumCircuit(5)
    theta_params = []

    # First layer: H + RY
    for i in range(5):
        quantum_circuit.h(i)
        theta = Parameter(f"θ_{len(theta_params)}")
        quantum_circuit.ry(theta, i)
        theta_params.append(theta)

    # CNOTs
    for i in range(4):
        quantum_circuit.cx(i, i + 1)

    quantum_circuit.barrier()

    # Second RY layer
    for i in range(5):
        theta = Parameter(f"θ_{len(theta_params)}")
        quantum_circuit.ry(theta, i)
        theta_params.append(theta)

    # CNOTs
    for i in range(4):
        quantum_circuit.cx(i, i + 1)

    quantum_circuit.barrier()

    # Third RY layer
    for i in range(5):
        theta = Parameter(f"θ_{len(theta_params)}")
        quantum_circuit.ry(theta, i)
        theta_params.append(theta)

    # Do not measure here; add measurements only in simulation
    return quantum_circuit, theta_params
# General imports
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt
# runtime imports
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from hamiltonian_library import

def run_on_hardware():
    # To run on hardware, select the backend with the fewest number of jobs in the queue
    raise Exception("Are you sure you want to run this on hardware? If so delete this exception from the code.")
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    return service, backend


def run_on_simulator():
    # To run on a simulator
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=True)
    return service, backend


def run_on_local_simulator():
    # Use local Aer simulator
    return None, AerSimulator()

if __name__ == '__main__':
    _, backend = run_on_local_simulator()
    hamiltonian =

def get_ansatz(number_of_qubits):
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

ansatz = EfficientSU2(hamiltonian.num_qubits)
ansatz.decompose().draw("mpl", style="iqp")
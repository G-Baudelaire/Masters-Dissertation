from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService


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
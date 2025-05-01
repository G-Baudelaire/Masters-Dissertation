from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

# QiskitRuntimeService.save_account(
#     channel="ibm_cloud",
#     token="61a882cee92f9fcc7dcdf85d571eed53adb8c74a1f58a13dba929a401528ce358bdb309e3505d198abb5d4f634db244f3a2c184f93b0ace487a89ad2cdc7bc51"    # paste the token you just copied
# )

from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

service = QiskitRuntimeService()

# Bell Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Specify a QPU to use for the noise model

real_backend = service.backend("ibm_brisbane")
noise_model = NoiseModel.from_backend(real_backend)
# aer = AerSimulator.from_backend(real_backend.)

print(noise_model)

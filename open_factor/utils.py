import math

from qiskit import transpile


class Utils:
    @staticmethod
    def bits_needed(n):
        bits_p = math.ceil(math.log2(math.sqrt(n)))
        bits_q = math.ceil(math.log2(n)) - 1
        return bits_p, bits_q

    @staticmethod
    def clean_bitstring_pair(bitstring, p_qubits, q_qubits):
        def shift_add_one(x):
            return (int(x, 2) << 1) + 1

        return shift_add_one(bitstring[:p_qubits]), shift_add_one(bitstring[-q_qubits:])

    @staticmethod
    def run_optimised_circuit(ansatz_isa, optimal_params, backend):
        bound_circuit = ansatz_isa.assign_parameters(optimal_params)
        bound_circuit.measure_all()
        compiled_circuit = transpile(bound_circuit, backend=backend)
        return backend.run(compiled_circuit, shots=1024).result()

    @staticmethod
    def check_bit_strings(n, bit_strings, p_qubits, q_qubits):
        pairs = (Utils.clean_bitstring_pair(bitstring, p_qubits, q_qubits) for bitstring in bit_strings)
        for p, q in pairs:
            if p * q == n:
                return p, q
        return None
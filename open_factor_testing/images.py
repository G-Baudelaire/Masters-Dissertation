from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, NLocal, RealAmplitudes, TwoLocal


def print_ansatz_images():
    # efficient_su = EfficientSU2(4, reps=2)
    # efficient_su.decompose().draw(output="mpl", scale=4.0).show()
    # real_amp = RealAmplitudes(4, reps=2)
    # real_amp.decompose().draw(output="mpl", scale=4.0).show()
    two_local = TwoLocal(4, reps=2, entanglement="linear")
    two_local.draw(output="mpl", scale=4.0).show()


def make_basic_circuit_image():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    qc.draw(output="mpl", scale=4.0).show()

    qc = QuantumCircuit(2)
    qc.y(0)
    qc.cx(0, 1)
    qc.measure_all()
    qc.draw(output="mpl", scale=4.0).show()


if __name__ == '__main__':
    print_ansatz_images()

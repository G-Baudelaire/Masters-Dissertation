from qiskit_aer import AerSimulator
from open_factor.sobhani_factoriser import SobhaniFactoriser

if __name__ == '__main__':
    backend = AerSimulator()
    print(SobhaniFactoriser(backend).factorise(77))
    # print(InverseFactoriser(backend).factorise(15))
    # logarithmic_factorise(1011)

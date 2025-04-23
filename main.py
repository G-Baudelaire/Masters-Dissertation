from project_code import Utils, run_on_local_simulator
from project_code.ansatzes import OptimisedLinearCNotAnsatz
from project_code.costs import CVaRCostFunction
from project_code.factorisation_methods import FactorisationMethods, InverseFactorisationMethod, \
    LogarithmFactorisationMethod
from project_code.hamiltonians import OptimisedSumOfSquaresHamiltonian


def run_hamiltonian_facorise(n):
    p_bits, q_bits = Utils.bits_needed(n)
    _, backend = run_on_local_simulator()

    FactorisationMethods.hamiltonian_factorise(
        n=n,
        ansatz_implementor=OptimisedLinearCNotAnsatz,
        hamiltonian_implementor=OptimisedSumOfSquaresHamiltonian,
        optimisation_implementor=CVaRCostFunction,
        attempts=100,
        backend=backend,
        p_bits=p_bits,
        q_bits=q_bits
    )


def inverse_factorise(n):
    p_bits, q_bits = Utils.bits_needed(n)
    _, backend = run_on_local_simulator()

    InverseFactorisationMethod.inverse_factorise(
        n=n,
        attempts=100,
        backend=backend,
        p_bits=p_bits,
        q_bits=q_bits
    )

def logarithmic_factorise(n):
    p_bits, q_bits = Utils.bits_needed(n)
    _, backend = run_on_local_simulator()

    LogarithmFactorisationMethod.inverse_factorise(
        n=n,
        attempts=100,
        backend=backend,
        p_bits=p_bits,
        q_bits=q_bits
    )


if __name__ == '__main__':
    logarithmic_factorise(1011)

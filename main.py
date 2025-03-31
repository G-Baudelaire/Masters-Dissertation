from project_code import Utils, run_on_local_simulator, factor
from project_code.ansatzes import OptimisedLinearCNotAnsatz
from project_code.costs import EstimatorOptimiser, CVaRCostFunction
from project_code.hamiltonians import OptimisedSumOfSquaresHamiltonian

if __name__ == '__main__':
    n = 11 * 7
    p_bits, q_bits = Utils.bits_needed(n)
    _, backend = run_on_local_simulator()

    # factorise(n, 100, backend, ansatz, hamiltonian, p_bits, q_bits)
    factor(
        n=n,
        ansatz_implementor=OptimisedLinearCNotAnsatz,
        hamiltonian_implementor=OptimisedSumOfSquaresHamiltonian,
        optimisation_implementor=CVaRCostFunction,
        attempts=100,
        backend=backend,
        p_bits=p_bits,
        q_bits=q_bits
    )

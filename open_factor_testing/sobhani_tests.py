from qiskit_aer import AerSimulator

from open_factor.ansatz_handlers import LinearCNOTAnsatzHandler
from open_factor.history_handlers import BlankHistoryHandler
from open_factor.sobhani_factoriser import SobhaniFactoriser, SobhaniOptimisationHandler


class TestSobhaniFactoriser:
    ATTEMPTS = 10

    def test_linear_cnot_cobyla(self):
        factoriser = SobhaniFactoriser(
            ansatz_handler=LinearCNOTAnsatzHandler(),
            optimisation_handler=SobhaniOptimisationHandler(),
            history_handler=BlankHistoryHandler(),
            backend=AerSimulator()
        )
        accuracy = self.run(factoriser, 77)
        print(accuracy)

    def run(self, factoriser, n):
        successes = sum(1 for _ in range(self.ATTEMPTS) if factoriser.factorise(n))
        return successes / self.ATTEMPTS


if __name__ == '__main__':
    TestSobhaniFactoriser().test_linear_cnot_cobyla()

from matplotlib import pyplot as plt


class HistoryHandler:
    def __init__(self):
        self._cost_history = None

    def set_history(self, cost_history) -> None:
        self._cost_history = cost_history

    def get_history(self) -> dict:
        return self._cost_history


class EnergyGrapherHistoryHandeler(HistoryHandler):
    def set_history(self, cost_history):
        fig, ax = plt.subplots()
        ax.plot(range(cost_history["iters"]), cost_history["cost_history"])
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        plt.show()


class BlankHistoryHandler(HistoryHandler):
    def set_history(self, cost_history) -> None:
        pass

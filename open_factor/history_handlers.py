from matplotlib import pyplot as plt


class HistoryHandler:
    def store_history(self, cost_history):
        raise NotImplementedError()


class SobhaniHistoryHandler(HistoryHandler):
    def store_history(self, cost_history):
        fig, ax = plt.subplots()
        ax.plot(range(cost_history["iters"]), cost_history["cost_history"])
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        plt.draw()

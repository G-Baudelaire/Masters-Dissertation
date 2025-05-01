import csv
import os.path

import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from matplotlib.pyplot import xscale
from qiskit import QuantumCircuit

from open_factor.ansatz_handlers import EfficentSU2AnsatzHandler
from open_factor.utils import Utils

import csv
import matplotlib.pyplot as plt


# Assuming EfficentSU2AnsatzHandler and Utils are already defined somewhere

def make_graph_of_efficient_su_squared_error_cobyla(path):
    with open(path, newline="") as f_in:
        reader = csv.DictReader(f_in)
        x, y = [], []

        for row in reader:
            if row["ansatz"] == EfficentSU2AnsatzHandler.__name__ and row["optimiser"] == "cobyla":
                x.append(sum(Utils.bits_needed(int(row["n"]))) - 2)
                y.append(float(row["accuracy"]))

        sorted_pairs = sorted(zip(x, y))
        x_sorted, y_sorted = zip(*sorted_pairs)

        plt.plot(x_sorted, y_sorted, marker="x")
        plt.yticks([i / 10 for i in range(0, 11)])  # y-axis from 0 to 1 in steps of 0.1
        plt.xticks(range(0, max(x) + 7, 5))  # x-axis from min(x) to max(x) with step 5
        plt.ylim(0, 1)
        plt.xlim(0, max(x) + 5)
        plt.xlabel("Qubits Needed")
        plt.ylabel("Accuracy")
        plt.title("EfficientSU2 Ansatz with COBYLA Optimiser")
        plt.grid(True)
        plt.show()
        print(x, y)


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


def draw_noiseless_baseline_graph():
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. Read the CSV
    df = pd.read_csv(os.path.join("results", "noiseless_baseline.csv"))

    # 2. Pivot to get one column per factoriser
    pivot = df.pivot(
        index="n",
        columns="factoriser",
        values="success_rate"
    )

    # 3. Plot directly
    ax = pivot.plot(
        kind="line",
        marker="o",
        figsize=(8, 5)
    )

    # 4. Labels & title
    ax.set_xlabel("n")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs n by Factoriser")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # make_graph_of_efficient_su_squared_error_cobyla(os.path.join("results", "accuracy_squared.csv"))
    draw_noiseless_baseline_graph()

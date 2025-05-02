import csv
import os.path

import matplotlib.pyplot as plt
import pandas as pd

from open_factor.ansatz_handlers import EfficentSU2AnsatzHandler
from open_factor.utils import Utils


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

import os
import pandas as pd
import matplotlib.pyplot as plt

def draw_shot_dependence_graph():
    # Load your data
    df = pd.read_csv(os.path.join("results", "shot_dependence.csv"))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot one line per 'n'
    for n_value, group in df.groupby('n'):
        group = group.sort_values('shots')
        ax.plot(
            group['shots'],
            group['success_rate'],
            marker='o',
            label=f"n = {n_value}"
        )

    # Logarithmic x‐axis that still allows zero
    # ax.set_xscale('symlog', linthresh=1)

    # Force origin at (0,0) and cap y at 1
    ax.set_xlim(left=0, right=df['shots'].max() * 1.1)
    ax.set_ylim(0, 1)

    ax.set_xlabel('Number of shots')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs. Number of Shots for Different $n$')
    ax.legend(title='Semiprime $n$')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def draw_ansatz_depth_graph():
    df = pd.read_csv(os.path.join("results", "ansatz_depth_comparison.csv"))

    fig, ax = plt.subplots(figsize=(8, 6))

    for (ansatz, reps), group in df.groupby(['ansatz', 'reps']):
        group = group.sort_values('n')
        ax.plot(
            group['n'],
            group['success_rate'],
            marker='o',
            label=f"{ansatz}, reps={reps}"
        )

    ax.set_xlabel('n')
    ax.set_ylabel('Success Rate')

    # use a logarithmic x‐axis
    ax.set_xscale('log')
    ax.set_xlim(left=df['n'].min(), right=df['n'].max())

    ax.set_ylim(bottom=0, top=1)

    ax.legend(title='Ansatz & Reps')
    ax.grid(True, linestyle='--', alpha=0.5)
    # get current limits
    xmin, xmax = ax.get_xlim()

    # bump the right edge up by one decade (×10)
    ax.set_xlim(xmin, xmax * 10)
    plt.tight_layout()
    plt.show()


def draw_noiseless_baseline_graph():
    df = pd.read_csv(os.path.join("results", "noiseless_baseline.csv"))

    pivot = df.pivot(
        index="n",
        columns="factoriser",
        values="success_rate"
    )

    ax = pivot.plot(
        kind="line",
        marker="o",
        figsize=(8, 5)
    )

    ax.set_xlabel("n")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs n by Factoriser")

    # use a logarithmic x‐axis
    ax.set_xscale('log')
    ax.set_xlim(left=df['n'].min(), right=df['n'].max())

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)
    # get current limits
    xmin, xmax = ax.get_xlim()

    # bump the right edge up by one decade (×10)
    ax.set_xlim(xmin, xmax * 10)
    plt.tight_layout()
    plt.show()



#
# def draw_noiseless_baseline_graph():
#     df = pd.read_csv(os.path.join("results", "noiseless_baseline.csv"))
#
#     # 2. Pivot to get one column per factoriser
#     pivot = df.pivot(
#         index="n",
#         columns="factoriser",
#         values="success_rate"
#     )
#
#     # 3. Plot directly
#     ax = pivot.plot(
#         kind="line",
#         marker="o",
#         figsize=(8, 5)
#     )
#
#     # 4. Labels & title
#     ax.set_xlabel("n")
#     ax.set_ylabel("Success Rate")
#     ax.set_title("Success Rate vs n by Factoriser")
#     ax.grid(True)
#
#     # 5. Force axes to start at (0, 0)
#     ax.set_xlim(left=0)
#     ax.set_ylim(bottom=0)
#
#     plt.tight_layout()
#     plt.show()
#
# def draw_ansatz_depth_graph():
#     df = pd.read_csv(os.path.join("results", "ansatz_depth_comparison.csv"))
#
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     # Group by ansatz and reps and plot each as a separate line
#     for (ansatz, reps), group in df.groupby(['ansatz', 'reps']):
#         group = group.sort_values('n')
#         ax.plot(
#             group['n'],
#             group['success_rate'],
#             marker='o',
#             label=f"{ansatz}, reps={reps}"
#         )
#
#     # Set axis labels
#     ax.set_xlabel('n')
#     ax.set_ylabel('Success Rate')
#
#     # Force axes to start at (0,0)
#     ax.set_xlim(left=0)
#     ax.set_ylim(bottom=0)
#
#     # Add legend and grid
#     ax.legend(title='Ansatz & Reps')
#     ax.grid(True, linestyle='--', alpha=0.5)
#
#     plt.tight_layout()
#     plt.show()

if __name__ == '__main__':
    # make_graph_of_efficient_su_squared_error_cobyla(os.path.join("results", "accuracy_squared.csv"))
    # draw_noiseless_baseline_graph()
    # draw_ansatz_depth_graph()
    draw_shot_dependence_graph()

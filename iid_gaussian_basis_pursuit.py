# -*- coding: utf-8 -*-
"""
Created on 11/20/25 12:06:18

@author: hmellow

In this example we examine the sharp phase transition of basis pursuit success with respect to the oversampling ratio
and sparsity level.

The standard form of basis pursuit is:
min_x ||x||_1
s.t. y=Ax, where y is observed, and we want to recover x
Essentially, we are looking for a sparse x satisfying the system

In this example we're interested in the case where the linear system is underdetermined.
Also, we examine the case where the entries of A are i.i.d. Gaussian.
"""
import numpy as np
import cvxpy as cp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_signal(p, sparsity_level):
    """

    :param p: Dimension
    :param sparsity_level: in [0, 1], percent of entries that are 0
    :return:
    """

    y = np.random.randn(p)
    num_zero_entries = math.floor(sparsity_level * p)
    zero_entries_indices = np.random.choice(p, num_zero_entries, replace=False)
    y[zero_entries_indices] = 0

    return y[:, np.newaxis]


OPTIMIZATION_TOLERANCE = 1e-6
RECOVERY_TOLERANCE = 1e-3


def run_basis_pursuit(p, n, sparsity_level):

    # Create A, x, y
    x_signal = generate_signal(p, sparsity_level)
    A = np.random.randn(n, p)
    y = A @ x_signal

    # Solve optimization problem
    x = cp.Variable(shape=(p, 1))
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.norm(A @ x - y) <= OPTIMIZATION_TOLERANCE]
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True, verbose=False)

    x_hat = x.value
    if x_hat is None:
        raise ValueError("x_hat is None")

    relative_error = np.linalg.norm(x_hat - x_signal) / np.linalg.norm(x_signal)
    if relative_error < RECOVERY_TOLERANCE:
        return True
    else:
        return False


def plot_scatter_results(delta_vals, s_vals, results):
    delta_grid, s_grid = np.meshgrid(delta_vals, s_vals, indexing="ij")
    colors = np.where(results, "blue", "red")

    delta_flat = delta_grid.flatten()
    s_flat = s_grid.flatten()
    colors_flat = colors.flatten()

    plt.figure()
    plt.scatter(delta_flat, s_flat, c=colors_flat)
    plt.xlabel("$\\delta=\\frac{n}{p}$ (oversampling ratio)")
    plt.ylabel("s (sparsity)")
    plt.title("Basis Pursuit Phase Transition ($A$ i.i.d Gaussian)")
    success_patch = mpatches.Patch(color="blue", label="Success")
    failure_patch = mpatches.Patch(color="red", label="Failure")
    plt.legend(handles=[success_patch, failure_patch], loc="best")
    plt.show()


def main():
    p = 1000
    num_data_points = 1000

    num_1d = math.floor(np.sqrt(num_data_points))
    delta_vals = np.linspace(0.01, 0.99, num_1d)
    s_vals = np.linspace(0.01, 0.99, num_1d)
    results = np.full((num_1d, num_1d), np.nan)
    p_bar = tqdm(total=num_data_points, desc="Running trials")

    for i, delta in enumerate(delta_vals):
        # delta = n/p (oversampling ratio)
        n = max(1, math.floor(delta * p))
        for j, s in enumerate(s_vals):
            results[i, j] = run_basis_pursuit(p, n, s)
            p_bar.update(1)
            p_bar.refresh()

    plot_scatter_results(delta_vals, s_vals, results)


if __name__ == "__main__":
    main()

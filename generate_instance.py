import numpy as np
import pandas as pd
import os
from utils import setup_logging


def generate_assignment_problem_instance(n, low=1, high=100, seed=None):
    """
    Generate an assignment problem instance as an n x n cost matrix.

    Args:
        n (int): The size of the problem (number of agents/tasks).
        low (int): The minimum cost value (inclusive).
        high (int): The maximum cost value (exclusive).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: An n x n numpy array representing the cost matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    cost_matrix = np.random.randint(low, high, size=(n, n))
    return cost_matrix


def generate_knapsack_problem_instance(n, weight_low=1, weight_high=50, value_low=1, value_high=100, capacity=500, seed=None):
    """
    Generate a knapsack problem instance.

    Args:
        n (int): Number of items.
        weight_low (int): Minimum item weight (inclusive).
        weight_high (int): Maximum item weight (exclusive).
        value_low (int): Minimum item value (inclusive).
        value_high (int): Maximum item value (exclusive).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: An n x 2 numpy array, each row is [weight, value].
        int: The maximum weight W.
    """
    if seed is not None:
        np.random.seed(seed)
    weights = np.random.randint(weight_low, weight_high, size=n)
    values = np.random.randint(value_low, value_high, size=n)
    items = np.stack([weights, values], axis=1)
    # Set W as about half the sum of all weights (classic setting)
    W = capacity
    return items, W


def generate_bin_packing_problem_instance(n, item_low=1, item_high=20, bin_capacity=50, seed=None):
    """
    Generate a bin packing problem instance.

    Args:
        n (int): Number of items.
        item_low (int): Minimum item size (inclusive).
        item_high (int): Maximum item size (exclusive).
        bin_capacity (int): Capacity of each bin.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: A 1D numpy array of item sizes.
        int: The bin capacity W.
    """
    if seed is not None:
        np.random.seed(seed)
    items = np.random.randint(item_low, item_high, size=n)
    return items, bin_capacity


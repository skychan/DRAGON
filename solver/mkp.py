"""Solves a multiple knapsack problem using the CP-SAT solver."""
from ortools.sat.python import cp_model
import json
import argparse


def solve_mkp(data, time_limit_sec=10):
    weights = data["weights"]
    values = data["values"]
    bin_capacities = data["capacity"]

    assert len(weights) == len(values)
    num_items = len(weights)
    all_items = range(num_items)
    num_bins = len(bin_capacities)
    all_bins = range(num_bins)

    model = cp_model.CpModel()

    # Variables.
    x = {}
    for i in all_items:
        for b in all_bins:
            x[i, b] = model.new_bool_var(f"x_{i}_{b}")

    # Constraints.
    for i in all_items:
        model.add_at_most_one(x[i, b] for b in all_bins)
    for b in all_bins:
        model.add(
            sum(x[i, b] * weights[i] for i in all_items) <= bin_capacities[b]
        )

    # Objective.
    objective = []
    for i in all_items:
        for b in all_bins:
            objective.append(cp_model.LinearExpr.term(x[i, b], values[i]))
    model.maximize(cp_model.LinearExpr.sum(objective))

    solver = cp_model.CpSolver()
    print(f"Time limit: {time_limit_sec} seconds")
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = args.num_worker
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Objective value: {solver.objective_value}")
        if status == cp_model.OPTIMAL:
            print("Get Optimal Value!")
            print(f"Relative gap: {0}")
            print(f"Bound: {solver.BestObjectiveBound()}")
        else:
            gap = abs(solver.BestObjectiveBound() - solver.ObjectiveValue()) / abs(solver.ObjectiveValue())
            print(f"Relative gap: {gap:.4%}")
            print(f"Bound: {solver.BestObjectiveBound()}")
        return solver.objective_value
    else:
        print("The problem does not have an optimal or feasible solution.")
        return None


if __name__ == "__main__":
    # Example usage
    # data = {
    #     "weights": [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36],
    #     "values": [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25],
    #     "bin_capacities": [100, 100, 100, 100, 100],
    # }

    parser = argparse.ArgumentParser(description="Solve a multiple knapsack problem.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input JSON file containing knapsack data.")
    parser.add_argument("--time_limit", type=int, default=10, help="Time limit for the solver in seconds.")
    parser.add_argument("--num_worker", type=int, default=1, help="Number of workers to use.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    solve_mkp(data, time_limit_sec=args.time_limit)
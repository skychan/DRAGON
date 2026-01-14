import io
import json
import argparse
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model



def solve_ap(data, time_limit_sec=10, num_worker=1):
    num_workers = data["num_workers"]
    costs = np.array(data["costs"], dtype=int)
    num_tasks = costs.shape[1]

    model = cp_model.CpModel()
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # Each worker assigned to at most one task
    for i in range(num_workers):
        model.AddAtMostOne(x[i, j] for j in range(num_tasks))
    # Each task assigned to exactly one worker
    for j in range(num_tasks):
        model.AddExactlyOne(x[i, j] for i in range(num_workers))

    # Objective: minimize total cost
    model.Minimize(
        sum(costs[i, j] * x[i, j] for i in range(num_workers) for j in range(num_tasks))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = num_worker

    print(f"Time limit: {time_limit_sec} seconds, num_worker: {num_worker}")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Objective value: {solver.ObjectiveValue()}")
        # Optionally, print assignment
        assignment = []
        for i in range(num_workers):
            for j in range(num_tasks):
                if solver.BooleanValue(x[i, j]):
                    assignment.append((i, j))
        print(f"Assignment: {assignment}")
        return solver.ObjectiveValue()
    else:
        print("The problem does not have an optimal or feasible solution.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve an assignment problem.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input JSON file containing assignment data.")
    parser.add_argument("--time_limit", type=int, default=10, help="Time limit for the solver in seconds.")
    parser.add_argument("--num_worker", type=int, default=1, help="Number of workers to use for the solver.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    solve_ap(data, time_limit_sec=args.time_limit, num_worker=args.num_worker)
from ortools.linear_solver import pywraplp
import argparse
import json


def solve_bpp(data, time_limit_sec=10):
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Solver not created.")
        return None

    # Set time limit (milliseconds)
    solver.SetTimeLimit(time_limit_sec * 1000)
    solver.EnableOutput()


    weights = data["items"]
    items = range(len(data["items"]))
    bin_capacity = data["capacity"]
    bins = list(range(len(items)))  # One bin per item (worst case)

    # Variables
    x = {}
    for i in items:
        for j in bins:
            x[(i, j)] = solver.IntVar(0, 1, f"x_{i}_{j}")

    y = {}
    for j in bins:
        y[j] = solver.IntVar(0, 1, f"y_{j}")

    # Constraints
    for i in items:
        solver.Add(sum(x[i, j] for j in bins) == 1)
    for j in bins:
        solver.Add(
            sum(x[i, j] * weights[i] for i in items) <= y[j] * bin_capacity
        )

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j] for j in bins]))

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        num_bins = 0
        solution = []
        for j in bins:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in items:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += weights[i]
                if bin_items:
                    num_bins += 1
                    solution.append({"bin": j, "items": bin_items, "weight": bin_weight})
        print(f"Number of bins used: {num_bins}")
        print(f"Objective value: {solver.Objective().Value()}")
        print(f"Solution: {solution}")
        print("Time = ", solver.WallTime(), " milliseconds")
        return num_bins
    else:
        print("The problem does not have an optimal or feasible solution.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve a bin packing problem."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing bin packing data.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=10,
        help="Time limit for the solver in seconds.",
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    solve_bpp(data, time_limit_sec=args.time_limit)

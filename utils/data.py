import pandas as pd
import numpy as np
import json


def read_co_problem_instance(problem_type='tsp', file_path = "", num_decimals=0, **kwargs):
    """
    Generalized CO problem reader. Handles in-memory or file-based TSP input.
    """

    if problem_type == "tsp":
        tsp_data = read_data_tsp(file_path, num_decimals=num_decimals, task_id=kwargs.get("task_id", -1))
        return {
            "node_coord": tsp_data["node_coord"],
        }

    if problem_type == "cvrp":
        cvrp_data = read_data_vrp(file_path, num_decimals=num_decimals)
        return {
            "type": cvrp_data["type"],
            "dimension": cvrp_data["dimension"],
            "edge_weight_type": cvrp_data["edge_weight_type"],
            "capacity": cvrp_data["capacity"],
            "node_coord": cvrp_data["node_coord"],
            "demand": cvrp_data["demand"],
            "edge_weight": cvrp_data["edge_weight"]
        }

    if problem_type == "knapsack":
        knapsack_data = read_data_knapsack(file_path)
        return {
            "name": knapsack_data["name"],
            "capacity": knapsack_data["capacity"],
            "num_items": knapsack_data["num_items"],
            "values": knapsack_data["values"],
            "weights": knapsack_data["weights"]
        }

    if problem_type == "assignment":
        assignment_data = read_data_assignment(file_path)
        return {
            "name": assignment_data["name"],
            "num_workers": assignment_data["num_workers"],
            "costs": assignment_data["costs"]
        }

    if problem_type == "bin_pack":
        name = file_path
        # task_id = kwargs.get("task_id", 'u500_00')
        bp_data = read_data_binpack(name)
        return {
            "capacity": bp_data["capacity"],
            "items": bp_data["items"],
            "num_items": bp_data["num_items"],
        }

    raise ValueError(f"Unsupported problem type: {problem_type}. Supported types are: tsp, cvrp, knapsack, assignment, bin_pack.")


def read_data_tsp(file_path, num_decimals=0, task_id=-1):
    """
    Reads TSP data from .csv, .npy, or .pkl file and returns coordinates.
    task_id = -1 means no specific task, otherwise it reads the specified task.
    Returns:
        node_coord: list of (x, y) tuples
    """
    if file_path.endswith('csv'):
        df = pd.read_csv(file_path)
        coords = (df[['x', 'y']] * (10**num_decimals)).astype(int).values.tolist()
        return {
            "node_coord": coords
        }

    if file_path.endswith('npy'):
        data = np.load(file_path)
        data = np.round(data * (10**num_decimals)).astype(int)
        coords = data[:, :2].tolist()
        return {
            "node_coord": coords
        }

    # Pickle assumed to be a list-of-tuples structure
    data = pd.read_pickle(file_path)
    coords = [(x, y) for x, y in data[task_id]]
    return {
        "node_coord": coords
    }


def read_data_vrp(file_path, num_decimals=0):
    """
    Reads VRP data from a .vrp file and returns coordinates and demands.
    Returns:
        type: like CVRP
        dimension: number of nodes
        edge_weight_type: like EUC_2D
        capacity: vehicle capacity
        node_coord: list of (x, y) tuples
        demand: list of demands for each node
        edge_weight: 2d array

    """
    import vrplib

    vrp_instance = vrplib.read_instance(file_path)
    coords = [(int(x * (10**num_decimals)), int(y * (10**num_decimals))) for x, y in vrp_instance['node_coord']]
    edge_weight = np.array(vrp_instance['edge_weight'])
    vrp_instance.update({
        "node_coord": coords,
        "edge_weight": edge_weight
    })
    print(len(vrp_instance['demand']), "nodes of demand")
    return vrp_instance


def read_data_binpack(name="OR3", task_id='u500_00'):
    """
    Reads Bin Packing data from a name, from 
    Returns:
        bin_packing_instance: dictionary with bin packing data
    """
    # from generate_bin_pack import datasets
    with open(name, 'r') as f:
        data = json.load(f)

    return {
        "capacity": data["capacity"],
        "items": data["items"],
        "num_items": data["num_items"],
    }


def read_data_knapsack(file_path):
    """
    Reads Knapsack data from a JSON file and returns the knapsack instance.
    Returns:
        knapsack_instance: dictionary with knapsack data
    """
    import json
    with open(file_path, "r") as f:
        data = json.load(f)
    # Ensure required fields exist
    required = ["name", "capacity", "num_items", "values", "weights"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key '{key}' in knapsack instance file: {file_path}")
    return {
        "name": data["name"],
        "capacity": data["capacity"],
        "num_items": data["num_items"],
        "values": data["values"],
        "weights": data["weights"],
    }


def read_data_assignment(file_path):
    """
    Reads Assignment data from a JSON file and returns the assignment instance.
    Returns:
        assignment_instance: dictionary with assignment data
    """
    import json
    with open(file_path, "r") as f:
        data = json.load(f)
    required = ["name", "num_workers", "task_id", "costs"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key '{key}' in assignment instance file: {file_path}")
    # costs = np.array(data["costs"], dtype=int)
    costs = data["costs"]
    return {
        "name": data["name"],
        "num_workers": data["num_workers"],
        "costs": costs,
    }


if __name__ == "__main__":
    # Example usage
    # TSP example
    tsp_file = 'instances/tsplib/berlin52.npy'
    print(read_co_problem_instance("tsp", tsp_file))

    # VRP example
    vrp_file = 'instances/CVRP/X/X-n101-k25.vrp'
    print(read_co_problem_instance("cvrp", vrp_file))

    # # Knapsack example
    # knapsack_file = 'instances/knapsack/knapsackA_10_50_0.json'
    # print(read_co_problem_instance("knapsack", knapsack_file))

    # ## Assignment example
    # assignment_file = 'instances/assignment/assignmentA_5_0.json'
    # print(read_co_problem_instance("assignment", assignment_file))

    # Bin Packing example
    # "O3" or "Weibull 5k"
    print(read_co_problem_instance("bin_pack", "OR3", task_id='u500_00'))


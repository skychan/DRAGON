from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd
# from utils import evaluate_distance


def create_data_model(locations, start, end):
    """Stores the data for the problem."""
    data = {}
    # Locations in block units
    data["locations"] = locations
    data["num_vehicles"] = 1
    data["start"] = [start]
    data['end'] = [end]
    return data


def compute_euclidean_distance_matrix(locations, num_decimals=0):
    """Returns a 2D NumPy array of scaled Euclidean distances between points."""
    locations = np.array(locations)
    diff = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    dist_matrix *= 10 ** num_decimals
    return np.round(dist_matrix).astype(int)


def read_data(file_path):
    '''
    Read instance from csv file

    Output:
        - location list
    '''
    df = pd.read_csv(file_path)
    return df.values, df.x.tolist(), df.y.tolist()


def print_solution_tsp(manager, routing, solution):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()}")
    index = routing.Start(0)
    plan_output = "Route:\n"
    route_distance = 0
    route = []
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        route.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    route.append(manager.IndexToNode(index))
    print(plan_output)
    plan_output += f"Objective: {route_distance}m\n"

    return [route]


def print_solution_cvrp(demands, num_vehicles, manager, routing, solution):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    routes = []
    for vehicle_id in range(num_vehicles):
        route = []
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            # if node_index not in dummy_node:
            route_load += demands[node_index]
            route.append(node_index)
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        route.append(manager.IndexToNode(index))
        routes.append(route)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    # print(f"Total load of all routes: {total_load}")
    return routes


def solve(problem: dict, num_decimals=0, quick=False, time_limit=60):
    if problem['problem_type'] == 'tsp':
        return solve_routing(problem['locations'], problem['start'], problem['end'], link=problem['link'], num_decimals=num_decimals, quick=quick, time_limit=time_limit)
    if problem['problem_type'] == 'cvrp':
        return solve_routing(problem['locations'], problem['depot'], vehicle_capacity=problem['capacity'], demands=problem['demands'], link=problem['link'], num_decimals=num_decimals, quick=quick, time_limit=time_limit)
    if problem['problem_type'] == 'knapsack':
        return solve_knapsack(problem['values'], problem['weights'], problem['capacity'])
    if problem['problem_type'] == 'assignment':
        return solve_assignment(problem['costs'])


def solve_assignment(costs, num_decimals=0):
    from ortools.linear_solver import pywraplp

    num_workers = len(costs)
    num_tasks = len(costs[0])

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Variables
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # print(f"Total cost = {solver.Objective().Value()}\n")
        result = []
        for i in range(num_workers):
            for j in range(num_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    # print(f"Worker {i} assigned to task {j}." + f" Cost: {costs[i][j]}")
                    result.append(j)
    else:
        print("No solution found.")

    assert len(result) == num_tasks == num_workers, "Assignment failed"
    assert max(result) == num_tasks - 1, "Assign wrong task"
    assert min(result) == 0, "Assign wrong task"
    assert len(set(result)) == num_tasks, "Assign duplicate task"

    obj_value = np.round(solver.Objective().Value(), num_decimals)

    return [result], obj_value


def solve_knapsack(values, weights, capacity):
    from ortools.algorithms.python import knapsack_solver
    # Create the solver.
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackSolver",
    )

    solver.init(values, [weights], [capacity])
    computed_value = solver.solve()

    packed_items = []
    packed_weights = []
    total_weight = 0
    print("Total value =", computed_value)
    for i in range(len(values)):
        if solver.best_solution_contains(i):
            packed_items.append(i)
            packed_weights.append(weights[i])
            total_weight += weights[i]
    print("Total weight:", total_weight)
    print("Packed items:", packed_items)
    print("Packed_weights:", packed_weights)

    return [packed_items]


def solve_routing(locations, start, end = None, vehicle_capacity = None, demands=None, link = None, num_decimals=0, quick=False, time_limit = 60):
    """Entry point of the program."""
    # # Instantiate the data problem.
    # data = create_data_model(locations, start, end)

    if demands is None:
        # tsp problem
        manager = pywrapcp.RoutingIndexManager(
            len(locations), 1, [start], [end]
        )
    else:
        # cvrp problem
        manager = pywrapcp.RoutingIndexManager(
            len(locations), len(vehicle_capacity), start
        )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(locations, num_decimals=num_decimals)
    max_dist = distance_matrix.max()
    factor = int(np.floor(np.log10(max_dist)))

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

        # if link is not None and ((from_node, to_node) in link or (to_node, from_node) in link):
        #     return 0

        # return distance_matrix[from_node][to_node] + int(10**factor + 0.5)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # selected_nodes = []
    # mute_nodes = []
    # for (s, e) in link:
    #     if s == start:
    #         selected_nodes.append(e)
    #     elif e == start:
    #         selected_nodes.append(s)
    #     else:
    #         mute_nodes.append((s,e))

    # for node in selected_nodes:
    #     node_index = manager.NodeToIndex(node)
    #     link_vars = []
    #     for vehicle_id in range(routing.vehicles()):
    #         start_idx = routing.Start(vehicle_id)
    #         end_idx = routing.End(vehicle_id)
    #         # Create boolean expressions for:
    #         # 1. depot -> node
    #         cond1 = routing.NextVar(start_idx) == node_index
    #         # 2. node -> depot
    #         cond2 = routing.NextVar(node_index) == end_idx

    #         # Collect the BoolVars
    #         link_vars.append(cond1.Var())
    #         link_vars.append(cond2.Var())

    #     # Add constraint: sum of depot->node and node->depot links across vehicles == 1
    #     routing.solver().Add(routing.solver().Sum(link_vars) == 1)


    # for (s,e) in mute_nodes:
    #     s_idx = manager.NodeToIndex(s)
    #     e_idx = manager.NodeToIndex(e)

    #     cond1 = routing.NextVar(s_idx) == e_idx
    #     cond2 = routing.NextVar(e_idx) == s_idx
    #     routing.solver().Add(routing.solver().Sum([cond1.Var(), cond2.Var()]) == 1)


    def process_selected_node(node):
        node_index = manager.NodeToIndex(node)
        link_vars = []
        for vehicle_id in range(routing.vehicles()):
            start_idx = routing.Start(vehicle_id)
            end_idx = routing.End(vehicle_id)

            cond1 = routing.NextVar(start_idx) == node_index
            cond2 = routing.NextVar(node_index) == end_idx

            link_vars.append(cond1.Var())
            link_vars.append(cond2.Var())

        routing.solver().Add(routing.solver().Sum(link_vars) == 1)

    def process_mute_link(s, e):
        s_idx = manager.NodeToIndex(s)
        e_idx = manager.NodeToIndex(e)

        cond1 = routing.NextVar(s_idx) == e_idx
        cond2 = routing.NextVar(e_idx) == s_idx

        routing.solver().Add(routing.solver().Sum([cond1.Var(), cond2.Var()]) == 1)

    for (s, e) in link:
        if s == start:
            process_selected_node(e)
        elif e == start:
            process_selected_node(s)
        else:
            process_mute_link(s, e)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        if from_node >= len(locations):
            return 0
        return demands[from_node]

    if demands is not None:
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            # [vehicle_capacity] * len(locations),  # vehicle maximum capacities
            vehicle_capacity,
            True,  # start cumul to zero
            "Capacity",
        )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # print("Time limit= ", time_limit)
    search_parameters.time_limit.seconds = time_limit
    # search_parameters.time_limit.FromSeconds(time_limit)
    if not quick:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit

    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        if demands is not None:
            result = print_solution_cvrp(demands, len(vehicle_capacity), manager, routing, solution)
        else:
            result = print_solution_tsp(manager, routing, solution)

    # print('=====Evaluate========')
    # X,Y = zip(*data['locations'])
    # print(evaluate_distance(X, Y, route, 3))
    # print(route)
    return result


if __name__ == "__main__":
    locations, X, Y = read_data('instances/TSP_10_0.csv')
    # print(locations)
    solve(locations, 3, 5, link= [(3,6)])
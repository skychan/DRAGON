from scipy.spatial.distance import pdist, squareform
from tspgrasp import (
    Grasp, CheapestArc, SemiGreedyArc, CheapestInsertion, SemiGreedyInsertion,
    LocalSearch, SimulatedAnnealing
)
import numpy as np
from utils import euclidean_distance
from utils import evaluate_objective, is_feasible_solution
from scipy.spatial import distance_matrix
import math


def heuristic_solution(points, x, y, link = None):
    # start, end = points[0], points[-1]
    # TSP
    start = points[0]
    X = np.array([[x[i], y[i]] for i in points])
    D = squareform(pdist(X))
    # D[start, end] = D[end, start] = -1

    if link is not None:
        for s, e in link:
            D[s, e] = D[e, s] = -1

    grasp = Grasp()
    sol = grasp(D).tour

    assert len(set(sol)) == len(points)
    assert len(sol[:-1]) == len(points)
    sol = sol[:-1]

    sid = sol.index(start)
    result = sol[sid:] + sol[:sid] + [start]

    return result


def random_insertion_heuristic(points, start, end, dist_matrix, short_cuts=None):
    # Initialize the tour with the start and end points
    tour = [start, end]

    # Create a set of remaining points to be visited (excluding start and end)
    remaining_points = set(points) - set(tour)
    
    # Function to calculate the cost of inserting a point into a tour at a specific position
    def insertion_cost(tour, point):
        min_cost = float('inf')
        best_position = None
        for i in range(len(tour) - 1):
            # cost = (euclidean_distance(x[tour[i]], y[tour[i]], x[point], y[point]) +
            #         euclidean_distance(x[point], y[point], x[tour[i+1]], y[tour[i+1]]) -
            #         euclidean_distance(x[tour[i]], y[tour[i]], x[tour[i+1]], y[tour[i+1]]))
            cost = dist_matrix[tour[i]][point] + dist_matrix[point][tour[i+1]] - dist_matrix[tour[i]][tour[i+1]]
            if cost < min_cost:
                min_cost = cost
                best_position = i + 1
        return min_cost, best_position
    
    # Insert remaining points into the tour
    while remaining_points:
        point = np.random.choice(list(remaining_points))
        _, position = insertion_cost(tour, point)
        tour.insert(position, point)
        remaining_points.remove(point)

        if short_cuts is not None and point in short_cuts:
            point = short_cuts[point]
            tour.insert(position+1, point)
            remaining_points.remove(point)

    return np.array(tour).tolist()


def heuristic_solver(data, num_decimals=0):
    problem_type = data['problem_type']
    if problem_type == 'tsp':
        locations = data['locations']
        start = data['start']
        end = data['end']
        link = data['link']
        short_cuts = {}
        for s, e in link:
            short_cuts[s] = e
            short_cuts[e] = s

        x, y = zip(*locations)
        coords = np.column_stack((x, y))
        dist_matrix = distance_matrix(coords, coords)
        for s, e in link:
            dist_matrix[s, e] = dist_matrix[e, s] = -1
        sol = random_insertion_heuristic(list(range(len(locations))), start, end, dist_matrix, short_cuts=short_cuts)
        return [sol]
    elif problem_type == 'cvrp':
        raise NotImplementedError


# TSP Heuristic, to generate initial solution, and return the objective value, running time.
def tsp_init(x, y, start=None, end=None, num_decimals=0):
    assert start is not None and end is not None, "Start and end points must be provided."
    is_feasible = False
    while not is_feasible:
        coords = np.column_stack((x, y))
        dist_matrix = distance_matrix(coords, coords)
        sol_init = random_insertion_heuristic(list(range(len(x))), start, end, dist_matrix)
        # 'tsp', [sol_init], start=0, end=0, num=len(x)+1
        data = {
            'problem_type': 'tsp',
            'sol': [sol_init],
            'x': x,
            'y': y,
        }
        is_feasible, err_msg = is_feasible_solution(data)
    obj_init = evaluate_objective(data, num_decimals)
    return sol_init, obj_init



class Customer:
    def __init__(self, id, x, y, demand):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand

def distance(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def solve_cvrp_greedy(customers, vehicle_capacity, depot):
    routes = []
    unvisited_customers = set(customers)
    unvisited_customers.remove(depot)

    while unvisited_customers:
        current_route = [depot]
        current_load = 0
        current_customer = depot

        while True:
            next_customer = None
            min_distance = float('inf')

            for customer in unvisited_customers:
                if current_load + customer.demand <= vehicle_capacity:
                    dist = distance(current_customer, customer)
                    if dist < min_distance:
                        min_distance = dist
                        next_customer = customer

            if next_customer:
                current_route.append(next_customer)
                current_load += next_customer.demand
                unvisited_customers.remove(next_customer)
                current_customer = next_customer
            else:
                current_route.append(depot)
                routes.append(current_route)
                break
    routes_id = []
    for route in routes:
        route_id = []
        for customer in route:
            route_id.append(customer.id)
        routes_id.append(route_id)

    return routes, routes_id


def eval_solution(routes):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += distance(route[i], route[i + 1])
    return total_distance

# CVRP Heuristic
def cvrp_init(x, y, demands, capacity, depot=0, num_decimals=0):
    '''
    Using sweep_clustering heuristic to generate initial solution for CVRP.
    Output: list of list
    '''
    is_feasible = False
    customers = []
    for i in range(len(x)):
        customers.append(Customer(i, x[i], y[i], demands[i]))

    while not is_feasible:
        # Generate initial solution using sweep clustering
        # routes, total_distance = sweep_clustering(x, y, demands, capacity, depot, num_decimals)
        routes_c, routes = solve_cvrp_greedy(customers, capacity, customers[depot])
        total_distance = eval_solution(routes_c)
        data = {
            'problem_type': 'cvrp',
            'sol': routes,
            'x': x,
            'y': y,
            'depot': depot,
            'demands': demands,
            'capacity': capacity,
        }
        is_feasible, error_msg = is_feasible_solution(data)
    return routes, total_distance


def sweep_clustering(x, y, demands, capacity, depot = 0, num_decimals=0):
    from math import atan2
    '''
    Using sweep_clustering heuristic to generate initial solution for CVRP.
    Output: list of list
    '''
    
    depot_x, depot_y = x[depot], y[depot]
    customers = []
    for i in range(1, len(x)):
        angle = atan2(y[i] - depot_y, x[i] - depot_x)  # 计算极角
        customers.append((i, x[i], y[i], demands[i], angle))
    
    # 按极角排序
    customers.sort(key=lambda c: c[4])
    routes = []
    current_route = [0]
    current_load = 0
    
    for i, _, _, d, _ in customers:
        if current_load + d <= capacity:
            current_route.append(i)
            current_load += d
        else:
            current_route.append(0)  # 关闭当前路径
            routes.append(current_route)
            current_route = [0, i]  # 开启新路径
            current_load = d
    
    if current_route:
        current_route.append(0)
        routes.append(current_route)

    # calc objective value
    data = {
            'problem_type': 'cvrp',
            'sol': routes,
            'x': x,
            'y': y,
        }
    total_distance = evaluate_objective(data, num_decimals)

    return routes, total_distance


# Knapsack Heuristic
def knapsack_init(values, weights, capacity, num_decimals=0):
    is_feasible = False

    while not is_feasible:
        # Calculate value-to-weight ratio for each item
        pack = [(i, values[i], weights[i]) for i in range(len(values))]
        
        # Sort items by value-to-weight ratio in descending order
        pack.sort(reverse=True, key=lambda x: x[1] / x[2])
        
        total_value = 0
        total_weight = 0
        selected_items = []
        
        # Iterate over sorted items and add them to the knapsack
        for i, value, weight in pack:
            if total_weight + weight <= capacity:
                selected_items.append(i)
                total_value += value
                total_weight += weight
        
        candidate_items = [item for item in range(len(values)) if item not in selected_items]
        data = {
            'problem_type': 'knapsack',
            'sol': [selected_items, candidate_items],
            'weights': weights,
            'capacity': capacity,
        }
        
        is_feasible, error_msg = is_feasible_solution(data)


    return [selected_items, candidate_items], total_value


def knapsack_init(values, weights, capacities, num_decimals=0):
    """
    A greedy heuristic for the Multiple Knapsack Problem.
    Uses a First Fit Decreasing (FFD) approach.

    Args:
        values (list or np.array): List of values for each item.
        weights (list or np.array): List of weights for each item.
        capacities (list or np.array): List of capacities for each knapsack.

    Returns:
        tuple: A tuple containing:
            - assignments_by_knapsack (list of lists): Each inner list contains
              the indices of items assigned to that knapsack.
            - total_value (float): The total value of all items assigned across all knapsacks.
    """
    num_items = len(values)
    num_knapsacks = len(capacities)

    # 1. Prepare items with their original index, value, and weight
    # We will sort this list based on value/weight ratio
    items_info = []
    for i in range(num_items):
        if weights[i] > 0: # Avoid division by zero
            items_info.append({'index': i, 'value': values[i], 'weight': weights[i], 'ratio': values[i] / weights[i]})
        else: # Handle zero weight items - typically these are always taken if value > 0
            items_info.append({'index': i, 'value': values[i], 'weight': weights[i], 'ratio': float('inf') if values[i] > 0 else -float('inf')})


    # 2. Sort items by value-to-weight ratio in descending order
    items_info.sort(key=lambda x: x['ratio'], reverse=True)

    # 3. Initialize knapsack states
    # current_knapsack_weights[k] will store the current weight in knapsack k
    current_knapsack_weights = [0.0] * num_knapsacks
    # assignments_by_knapsack[k] will store a list of item indices assigned to knapsack k
    assignments_by_knapsack = [[] for _ in range(num_knapsacks)]
    total_overall_value = 0.0

    # 4. Iterate through sorted items and assign them to knapsacks
    for item in items_info:
        item_idx = item['index']
        item_value = item['value']
        item_weight = item['weight']

        # Try to fit the item into the first knapsack that can accommodate it (First Fit)
        assigned_to_knapsack = False
        for k_idx in range(num_knapsacks):
            if current_knapsack_weights[k_idx] + item_weight <= capacities[k_idx]:
                # Assign item to this knapsack
                assignments_by_knapsack[k_idx].append(item_idx)
                current_knapsack_weights[k_idx] += item_weight
                total_overall_value += item_value
                assigned_to_knapsack = True
                break # Move to the next item

        # If the item could not be assigned to any knapsack, it is left out
        if not assigned_to_knapsack:
            # print(f"Item {item_idx} (weight {item_weight}) could not fit into any knapsack.")
            pass # Item is simply not included in the solution

    # Check feasibility (optional, but good for robust code)
    # The heuristic's construction ensures feasibility for assigned items.
    data = {
        'problem_type': 'knapsack',
        'sol': assignments_by_knapsack,
        'values': values,
        'weights': weights,
        'capacity': capacities,
    }
    is_feasible, error_msg = is_feasible_solution(data)
    if not is_feasible:
        print(f"Warning: Heuristic generated an infeasible solution: {error_msg}")
        # This should ideally not happen if the logic is correct

    # Construct the format similar to your original output for unselected items
    all_selected_indices = []
    for k_assign in assignments_by_knapsack:
        all_selected_indices.extend(k_assign)
        
    candidate_items = [i for i in range(num_items) if i not in all_selected_indices]
    
    # Return in a structured way that includes assignments per knapsack
    return {
        'total_value': total_overall_value,
        'assignments_by_knapsack': assignments_by_knapsack, # List of lists of item indices
        'all_selected_items': all_selected_indices, # All items selected overall
        'unselected_items': candidate_items # Items that were not placed in any knapsack
    }


# Bin Packing Heuristic
def bin_pack_init(items, capacity, num_decimals=0):
    """
    Greedy First-Fit Decreasing heuristic for the bin packing problem.
    
    Args:
        items (list of float): List of item sizes.
        capacity (float): Bin capacity.
        num_decimals (int): Number of decimals to round for numerical stability (optional).
    
    Returns:
        Tuple[List[List[float]], int]: (Packed bins as lists of item sizes, number of bins used)
    """
    # Optional rounding to avoid floating point issues
    if num_decimals > 0:
        items = [round(item, num_decimals) for item in items]
        capacity = round(capacity, num_decimals)
    
    # Sort items in decreasing order
    indexes = range(len(items))
    sorted_items = sorted(indexes, key=lambda i: items[i], reverse=True)

    bins = []  # list of bins, each bin is a list of items
    for idx in sorted_items:
        placed = False
        # Try to place the item in an existing bin
        for bin in bins:
            if sum([items[i] for i in bin]) + items[idx] <= capacity:
                bin.append(idx)
                placed = True
                break
        # If not placed, create a new bin
        if not placed:
            bins.append([idx])

    data = {
        'problem_type': 'bin_pack',
        'sol': bins,
        'items': items,
        'capacity': capacity
    }
    is_feasible, error_msg = is_feasible_solution(data)
    if not is_feasible:
        print(f"Warning: Heuristic generated an infeasible solution: {error_msg}")

    num_bins_used = len(bins)
    return bins, num_bins_used



# Assignment Heuristic
def assignment_init(costs, num_decimals=0):
    # n = costs.shape[0]
    n = len(costs)
    assigned_agents = set()
    assigned_tasks = set()
    assignments = []
    total_cost = 0

    while len(assignments) < n:
        min_cost = float('inf')
        min_pair = (-1, -1)
        for i in range(n):
            if i in assigned_agents:
                continue
            for j in range(n):
                if j in assigned_tasks:
                    continue
                if costs[i][j] < min_cost:
                    min_cost = costs[i][j]
                    min_pair = (i, j)

        i, j = min_pair
        assignments.append((i, j))
        total_cost += costs[i][j]
        assigned_agents.add(i)
        assigned_tasks.add(j)
    
    # Round the total cost to the specified number of decimals
    if num_decimals > 0:
        total_cost = round(total_cost, num_decimals)
    # convert assignment to dict
    assignments = {i: j for i, j in assignments}
    # Convert assignments to a list of tuples
    result = [assignments[i] for i in range(n)]

    assert len(set(result)) == n, "Assignments should be unique."
    assert min(result) == 0 and max(result) == n - 1, "Assignments should be in the range [0, n-1]."
    
    return [result], total_cost


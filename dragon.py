from solver import orsolver, heuristic_solver, llm_solver, meta_partition
import os
import numpy as np
import pandas as pd
import pickle
import time
import json
from utils.metrics import Metrics
from utils import read_co_problem_instance, plot_route, evaluate_objective
from utils import point_rename, is_feasible_metaroute, is_feasible_solution, recover_routes, temperature_schedule, accept
from solver import tsp_init, cvrp_init, assignment_init, knapsack_init, bin_pack_init
from itertools import chain


def solve_subproblem(problem, method='random', save_root=None, epoch=0, n_examples = None, num_decimals = 0, metric : Metrics = None):
    '''
    Input:
        problem: dict, contains:
            - problem_type: str, 'tsp', 'cvrp', 'knapsack', 'assignment', 'bin_pack'
            - active_zone: tuple, (meta_points, meta_route)
            - static_zone: list of short cuts, [[1,2], [3,4], ...]
            - meta_sol: (optional for routing) list of point index, include end
            - x, y: all x and y coordinates
        - use_llm: bool
    Output:
        - route: list of point index, include end
    '''
    # problem transformation
    problem_type = problem['problem_type']
    if problem_type == 'tsp':
        meta_route = problem['meta_sol']
        x, y = problem['x'], problem['y']
        point_coord, p_map, p_map_rev = point_rename(meta_route, x, y)

        link = [(p_map_rev[zone[0]], p_map_rev[zone[-1]]) for zone in problem['static_zone']]

        locations = [point_coord[i] for i in range(len(point_coord))]
    elif problem_type == 'cvrp':
        x, y = problem['x'], problem['y']
        capacity = problem['capacity']
        meta_route = problem['meta_sol']
        depot = meta_route[0][0]
        # depot = 0
        all_nodes = list(set(chain(*meta_route)))

        point_coord, p_map, p_map_rev = point_rename(all_nodes, x, y)

        link = [(p_map_rev[zone[0]], p_map_rev[zone[-1]]) for zone in problem['static_zone'] if zone[0] != zone[-1]]

        # convert meta_route
        meta_sol = [
            [p_map_rev[node] for node in route] for route in meta_route
        ]
        locations = [point_coord[i] for i in range(len(point_coord))]

        # re-assign demand
        demand = [0] * len(point_coord)
        for node in point_coord:
            demand[node] = problem['demands'][p_map[node]]
        m_depot = p_map_rev[depot]
        for zone in problem['static_zone']:
            if zone[0] == zone[-1]:
                continue
            sum_demand = sum(problem['demands'][node] for node in zone)
            a, b = p_map_rev[zone[0]], p_map_rev[zone[-1]]
            if b == m_depot:
                demand[a], demand[b] = sum_demand, 0
                continue
            demand[a], demand[b] = 0, sum_demand
    elif problem_type == 'knapsack':
        static_zone = problem['static_zone']
        active_zone = problem['active_zone']
        capacity = problem['capacity']

        remain_capacities = [cap - sum(problem['weights'][item] for item in items) for items, cap in zip(static_zone, capacity)]

        positive_knapsacks = [idx for idx, cap in enumerate(remain_capacities) if cap > 0]
        k_map = {i: item for i, item in enumerate(positive_knapsacks)}
        k_map_rev = {item: i for i, item in k_map.items()}
        positive_remain_capacities = [cap for cap in remain_capacities if cap > 0]

        # active_zone = active_zone + list(un_selected_items)

        # rename
        p_map = {i: item for i, item in enumerate(chain(*active_zone))}
        p_map_rev = {item: i for i, item in p_map.items()}
        weights = [problem['weights'][p_map[i]] for i in range(len(p_map))]
        values = [problem['values'][p_map[i]] for i in range(len(p_map))]
    elif problem_type == 'assignment':
        static_zone = problem['static_zone']
        active_zone = problem['active_zone']

        # remane
        p_map = {i: item for i, item in enumerate(chain(*active_zone))}
        p_map_rev = {item: i for i, item in p_map.items()}
    elif problem_type == 'bin_pack':
        static_zone = problem['static_zone']
        active_zone = problem['active_zone']
        capacity = problem['capacity']

        min_active_weight = min(problem['items'][i] for i in chain(*active_zone))

        used_capacities = [sum(problem['items'][item] for item in items) for items in static_zone]

        positive_bins = [idx for idx, used in enumerate(used_capacities) if used + min_active_weight <= capacity]
        new_weights = [used for used in used_capacities if used + min_active_weight < capacity]
        bag_idxes = [[i for i in static_zone[k]] for k in positive_bins]
        p_map = {i: item for i, item in enumerate(chain(range(len(new_weights)), *active_zone))}
        p_map_rev = {item: i for i, item in p_map.items()}
        weights = new_weights + [problem['items'][p_map[i]] for i in range(len(new_weights), len(new_weights) + len(active_zone))]

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}. Supported types are: tsp, cvrp.")

    # prepare data
    data = {
        'problem_type': problem_type
    }
    if problem_type == 'tsp':
        data['locations'] = locations
        x, y = zip(*locations)
        data['x'] = x
        data['y'] = y
        data['start'] = 0
        data['end'] = 0
        data['link'] = link
    elif problem_type == 'cvrp':
        data['locations'] = locations
        x, y = zip(*locations)
        data['x'] = x
        data['y'] = y
        data['depot'] = m_depot
        data['capacity'] = [capacity]*len(meta_sol) if method != 'llm' else capacity
        data['demands'] = demand
        data['metasol'] = meta_sol
        data['link'] = link
    elif problem_type == 'knapsack':
        data['values'] = values
        data['weights'] = weights
        data['capacity'] = positive_remain_capacities
    elif problem_type == 'bin_pack':
        data['items'] = weights
        data['capacity'] = capacity
    '''
     1. Solve by...
    '''
    if method == 'llm':
        result = llm_solver(data, num_decimals=num_decimals, save_root=save_root, epoch=epoch, metric=metric)
    elif method == 'solver':
        # OR Tools
        result = orsolver(data, num_decimals=num_decimals, quick=False, time_limit=100)
    elif method == 'heuristic':
        result = heuristic_solver(data, num_decimals=num_decimals)
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods are: llm, solver, heuristic.")

    '''
     2. Finalize results
    '''
    if problem_type in ['tsp', 'cvrp']:
        if not is_feasible_metaroute(result, link):
            # assert len(result) == len(meta_route)
            # assert len(set(result)) == len(set(meta_route))
            # recovery = [p_map[i] for i in result]
            # assert len(set(recovery)) == len(set(meta_route))
            return [], None
        result_rev = [[p_map[i] for i in res] for res in result]
        
        sol_repaired = recover_routes(result_rev, problem['static_zone'])
        return sol_repaired, result_rev

    if problem_type == 'knapsack':
        result_rev = [[] for _ in remain_capacities]

        for idx, assign in enumerate(result):
            result_rev[k_map[idx]] = [p_map[i] for i in assign]

        # result_rev = [[p_map[i] for i in res] for res in result]
        sol_repaired = [res + stat for res, stat in zip(result_rev, problem['static_zone'])]

        # if problem_type == 'knapsack':
        #     all_elements = set(chain(*problem['sol_prev']))
        #     sol_repaired.append(list(all_elements - set(sol_repaired[0])))

        # map back to the original style
        return sol_repaired, result_rev

    if problem_type == 'bin_pack':
        result_rev = [[p_map[i] for i in res] for res in result]
        sol_repaired = [zone for i, zone in enumerate(problem['static_zone']) if i not in positive_bins]
        for assign in result:
            repair = []
            for idx in assign:
                if idx < len(new_weights):
                    repair.extend(bag_idxes[idx])
                else:
                    repair.append(p_map[idx])
            sol_repaired.append(repair)
        return sol_repaired, result_rev

    return [], None


def solve(instance = 'instances/TSP_50_0.csv', logger = None, task_id = 0, num_iter = 10, reviser_size = 10, num_decimals = 0, resume_epoch = None, destory_method = 'llm', repair_method = 'llm', model = 'gpt-4o', init_temperature = 1.0, cooling_rate = 0.95, problem_category = 'tsp', opt_val=0.0, exp_name = '', use_ci=1):
    '''
    Input:
        - instance: filename or instance name for assignment problem.
        - task_id: task id
        - num_iter: total number of iteration for each task
        - reviser_size: reviser size
        - num_decimals: int
        - use_llm: llm for partition or not
    '''
    if task_id == -1:
        save_root = os.path.join('results', exp_name, problem_category, f"{model}_{use_ci}", instance.split('/')[-1].split('.')[0])
    else:
        save_root = os.path.join('results', exp_name, problem_category, f"{model}_{use_ci}", instance.split('/')[-1].split('.')[0], f"task_{task_id}")
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, 'solution'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'dialogs'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'visu'), exist_ok=True)

    logger.info(f'Using GPT model: {model} with ci = {use_ci}')

    metric = Metrics(model=model, use_ci=use_ci)
    '''
    Read instance
    '''

    data = read_co_problem_instance(problem_type=problem_category, file_path=instance, task_id=task_id, num_decimals=num_decimals)

    # unpack data
    if problem_category == 'tsp':
        X, Y = zip(*data['node_coord'])
        # normalize
        # max_coord = max(max(X), max(Y))
        # coord_data = {
        #     'x': [int(x / max_coord * (10 ** num_decimals)) for x in X],
        #     'y': [int(y / max_coord * (10 ** num_decimals)) for y in Y]
        # }
        problem_data = {
            'x': X,
            'y': Y,
        }
        with open(os.path.join(save_root, 'problem.json'), 'w') as f:
            json.dump(problem_data, f)
    elif problem_category == 'cvrp':
        X, Y = zip(*data['node_coord'])
        demand = data['demand']
        capacity = data['capacity']
        edge_weight = data['edge_weight']
        problem_data = {
            'x': X,
            'y': Y,
            'demands': demand.tolist(),
            'capacity': capacity,
            'depot': data.get('depot', 0),
        }
        with open(os.path.join(save_root, 'problem.json'), 'w') as f:
            json.dump(problem_data, f)
    elif problem_category == 'knapsack':
        capacity = data['capacity']
        values = data['values']
        weights = data['weights']
        problem_data = {
            'capacity': capacity,
            'values': values,
            'weights': weights,
        }
        with open(os.path.join(save_root, 'problem.json'), 'w') as f:
            json.dump(problem_data, f)
    elif problem_category == 'assignment':
        num_workers = data['num_workers']
        costs = data['costs']
        problem_data = {
            'num_workers': num_workers,
            'costs': costs,
        }
        with open(os.path.join(save_root, 'problem.json'), 'w') as f:
            json.dump(problem_data, f)
    elif problem_category == 'bin_pack':
        capacity = data['capacity']
        items = data['items']
        problem_data = {
            'capacity': capacity,
            'items': items,
        }
        with open(os.path.join(save_root, 'problem.json'), 'w') as f:
            json.dump(problem_data, f)
    else:
        raise ValueError(f"Unsupported problem category: {problem_category}. Supported categories are: tsp, cvrp, knapsack, assignment, bin_pack.")

    problem_data['problem_type'] = problem_category
    checkpoint = {}
    
    '''
    Solve by ortools, as the oracle
    '''
    # mip_result = orsolver(locations, 0, 0, num_decimals=num_decimals, quick=True)
    # dist = evaluate_distance(X, Y, mip_result, num_decimals)
    # plot_route(x, y, mip_result, dist, show_node_name=True, save_path=os.path.join(save_root, f'{instance.split("/")[1].split(".csv")[0]}_mip.png'))
    # checkpoint['mip_result'] = mip_result
    # checkpoint['mip_dist'] = dist

    '''
    Initialization
    '''
    start_time = time.time()
    sol_init = None
    obj_init = None

    if problem_category == 'tsp':
        sol_init, obj_init = tsp_init(X, Y, start=0, end=0, num_decimals=num_decimals)
        plot_route(X, Y, sol_init, obj_init, show_node_name=True,
                    save_path=os.path.join(save_root, 'visu', f'{instance.split("/")[1].split(".")[0]}_init_ri.png'))
    elif problem_category == 'cvrp':
        sol_init, obj_init = cvrp_init(X, Y, demand, capacity, depot=0, num_decimals=num_decimals)
        logger.info(f"Initial solution (CVRP routes): {sol_init}, objective: {obj_init}")
    elif problem_category == 'knapsack':
        # raise NotImplementedError("Knapsack problem initialization is not implemented yet.")
        data_init = knapsack_init(values, weights, capacity, num_decimals=num_decimals)
        sol_init, obj_init = data_init['assignments_by_knapsack'], -data_init['total_value']
        logger.info(f"Initial solution (knapsack): {sol_init}, objective (value): {obj_init}")
    elif problem_category == 'assignment':
        sol_init, obj_init = assignment_init(costs, num_decimals=num_decimals)
        logger.info(f"Initial solution (assignment): {sol_init}, objective (cost): {obj_init}")
    elif problem_category == 'bin_pack':
        sol_init, obj_init = bin_pack_init(items, capacity, num_decimals=num_decimals)
        logger.info(f"Initial solution (bin packing): {sol_init}, objective (bins used): {obj_init}")
    else:
        raise ValueError(f"Unsupported problem category: {problem_category}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Initialization: {elapsed_time:.6f} seconds")

    # Save/checkpoint using consistent keys with _init suffix
    checkpoint['initial_solution'] = sol_init
    checkpoint['initial_objective'] = obj_init

    sol_best = sol_init
    obj_best = obj_init
    checkpoint['best_solution'] = sol_best
    checkpoint['best_objective'] = obj_best
    checkpoint['best_iter'] = 0
    reject_count = 0
    checkpoint['reject_count'] = reject_count

    sol = sol_init
    obj = obj_init

    no_improve_count = 0
    problem_data['sol_prev'] = sol_init
    problem_data['obj_prev'] = obj_init
    for i in range(num_iter):
        # if resume_epoch is not None and i < resume_epoch:
        #     continue
        if no_improve_count > 5:
            logger.info(f"No improvement in 5 consecutive iterations, stopping early at iteration {i}.")
            break

        # partition
        logger.info(f"Start destory")
        start_time = time.time()

        '''
        Divide and Conqure (Sampled Sub-Problem)
        '''
        # use active and static zone in the future to be general
        active_zone, static_zone, meta_sol = meta_partition(sol, problem_type=problem_category, reviser_size=reviser_size, method=destory_method, save_root=save_root, epoch=i, reject_count = reject_count, metric=metric)

        end_time = time.time()
        logger.info(f"Partition Finished, using {end_time - start_time:.6f} seconds")

        # dist = evaluate_distance(x, y, route, num_decimals)
        # conqure
        logger.info(f"Start repair")
        logger.info(f"Iter {i}, obj = {obj}")
        start_time = time.time()

        problem_data['active_zone'] = active_zone
        problem_data['static_zone'] = static_zone
        problem_data['meta_sol'] = meta_sol

        revised, revised_meta_route = solve_subproblem(problem_data, num_decimals=num_decimals,save_root=save_root, epoch=i, method=repair_method, metric=metric) # include end point
        problem_data['sol'] = revised

        if len(revised) == 0:
            logger.info(f"Conqure Failed, infeasible sub-solution, using {end_time - start_time:.6f} seconds")
            no_improve_count += 1
            continue

        is_feasible, err_msg = is_feasible_solution(problem_data)

        if not is_feasible:
            # logger.info(f"Error: {err_msg}")
            # with open(os.path.join(save_root, 'error.pkl'), 'wb') as f:
            #     pickle.dump(problem_data, f)
            # return
            no_improve_count += 1
            continue

        end_time = time.time()

        obj_new = evaluate_objective(problem_data, num_decimals)
        if problem_category == 'knapsack':
            obj_new = -obj_new
        problem_data['obj'] = obj_new

        temperature = temperature_schedule(init_temperature, i, cooling_rate)
        # Add eplison greedy/accept function
        accepted = False
        if accept(obj, obj_new, temperature):
            if obj_new == obj:
                reject_count += 1
            else:
                reject_count = 0
                accepted = True
            if obj_new < obj:
                no_improve_count = 0
            else:
                no_improve_count += 1
            logger.info(f"Accept new solution, update route, rate = {np.exp((obj - obj_new) / temperature) if obj_new > obj else 1}")
            logger.info(f"Old: {obj} -> New: {obj_new}")
            # Update
            sol, sol_prev = revised, sol
            obj, obj_prev = obj_new, obj
            # Plot
            # plot_route(x, y, route, dist, before=route_prev, dist_old=dist_prev, link=meta_points, show_node_name=True, save_path=os.path.join(save_root, f'{instance.split("/")[1].split(".")[0]}_{method}_iter_{str(i).zfill(2)}.png'))
            # plot_route(x, y, route, dist, show_node_name=True, save_path=os.path.join(save_root, f'{instance.split("/")[1].split(".")[0]}_{method}.png'))
            if obj < obj_best:
                obj_best = obj
                sol_best = sol
                checkpoint['best_solution'] = sol_best
                checkpoint['best_objective'] = obj_best
                checkpoint['best_iter'] = i
                # plot_route(x, y, best_route, best_dist, show_node_name=True, save_path=os.path.join(save_root, f'{instance.split("/")[1].split(".")[0]}_{method}_best.png'))
        else:
            logger.info(f"Reject new solution, rate = {np.exp((obj - obj_new) / temperature)}")
            logger.info(f"Old: {obj} x->| New: {obj_new}")
            sol, sol_prev = sol, sol
            obj, obj_prev = obj, obj
            reject_count += 1
            no_improve_count += 1

        logger.info(f"Iter {i} (repaired): {obj}")
        logger.info(f"Route: {sol}")

        logger.info(f"Conqure Finished, using {end_time - start_time:.6f} seconds")
        checkpoint['obj'] = obj
        checkpoint['sol'] = sol
        checkpoint['reject_count'] = reject_count
        with open(os.path.join(save_root, 'checkpoint.json'), 'w') as f:
            json.dump(checkpoint, f)

        problem_data['iter'] = i
        problem_data['accepted'] = accepted

        os.makedirs(os.path.join('experiments', exp_name, problem_category, f"{model}_{use_ci}"), exist_ok=True)

        with open(os.path.join(save_root, 'solution', f'iter_{i:03d}.json'), 'w') as f:
            json.dump(problem_data, f)
        with open(os.path.join('experiments', exp_name, problem_category, f"{model}_{use_ci}", f"{os.path.basename(instance).split('.')[0]}.sol"), 'w') as f:
            json.dump(problem_data, f)
        
        problem_data['sol_prev'] = sol
        problem_data['obj_prev'] = obj

        metric_save_path = os.path.join('experiments', exp_name, problem_category, f"{model}_{use_ci}", 'experiments.csv')
        if problem_category in ['knapsack', 'bin_pack']:
            opt_val = obj
        metric.save_csv(metric_save_path, os.path.basename(instance).split('.')[0], obj, opt_val)


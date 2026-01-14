import numpy as np
import re
import os
from collections import Counter
from itertools import chain


def temperature_schedule(initial_temp, step, cooling_rate):
    """
    Calculates the temperature for the current step.

    Parameters:
    - initial_temp (float): The starting temperature.
    - step (int): The current step or iteration.
    - cooling_rate (float): The rate at which the temperature decreases.

    Returns:
    - float: The current temperature.
    """
    return initial_temp * (cooling_rate ** step)



def accept(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return True
    else:
        probability = np.exp((old_cost - new_cost) / temperature)
        return probability > np.random.rand()


def evaluate_objective(data, num_decimals):
    '''
        if loop trace, the end shoud be the start
    '''
    if data['problem_type'] == 'knapsack':
        values = data['values']
        total_values = sum(values[i] for i in chain(*data['sol']))
        return total_values
    if data['problem_type'] == 'bin_pack':
        return len(data['sol'])
    dis = 0
    routes = data['sol']
    link = data.get('link', None)
    x, y = data['x'], data['y']
    try:
        for route in routes:
            for head, tail in zip(route, route[1:]):
                if link is not None and ((head, tail) in link or (tail, head) in link):
                    continue
                dis += np.sqrt((x[head] - x[tail]) ** 2 + (y[head] - y[tail]) ** 2)
    except:
        return -1
    dis = np.round(dis, num_decimals) if num_decimals > 0 else int(dis)

    return dis


def point_reindex(points, x, y, start, end):
    all_points = [start] + points
    if end != start:
        all_points += [end]
    p_map = {i: p for i,p in enumerate(all_points)}
    p_map_rev = {p: i for i,p in p_map.items()}
    point_coord = {i: (x[p_map[i]], y[p_map[i]]) for i in p_map}

    return point_coord, p_map, p_map_rev


def point_rename(points, x, y):
    # cannot loop
    is_loop = points[0] == points[-1]
    if is_loop:
        raise NotImplementedError
    p_map = {i: p for i,p in enumerate(points)}
        # p_map_rev = {p: i for i,p in p_map.items() if i != len(points) - 1}
    # else:
    p_map_rev = {p: i for i,p in p_map.items()}
    point_coord = {i: (x[p_map[i]], y[p_map[i]]) for i in p_map}

    return point_coord, p_map, p_map_rev


def string_trace(trace, length):
    result = '<trace>'
    result += ','.join(str(p) for p in trace)
    result += '</trace>\n'
    result += f'<length>{length}</length>'
    return result


def extract_string(input_string):
    start_string = "<trace>"
    end_string = "</trace>"
    if start_string not in input_string:
        return ""
    input_string = input_string[input_string.index(start_string) + len(start_string):]
    if end_string not in input_string:
        return ""
    input_string = input_string[:input_string.index(end_string)]
    parsed_list = []
    for p in input_string.split(","):
        p = p.strip()
        try:
            p = int(p)
        except:
            continue
        parsed_list.append(p)
    return parsed_list


def extract_route_from_string(text):
    # Match start and end tags like <Route> ... </route> (case-insensitive)
    route_blocks = re.findall(
        r"<(?i:route)>\s*(.*?)\s*</(?i:route)>",
        text,
        flags=re.DOTALL
    )

    routes = []
    for block in route_blocks:
        # Extract all integers from the block, ignoring delimiters/wrappers
        numbers = list(map(int, re.findall(r"\d+", block)))
        routes.append(numbers)

    return routes

def extract_assignment_from_string(text):
    """
    Robustly extracts lists of integers from <knapsack_i> tags within <sol>...</sol> block.
    The tag name varies by row index (e.g., <knapsack_0>, <knapsack_1>, etc.).
    Handles various capitalization, potential attributes in tags, common typos,
    and self-closing tags like <knapsack_0/>.
    Corrects issue with `re.findall` consuming too much with `DOTALL` for inner matches.

    Args:
        text (str): The input string to search.

    Returns:
        list: A list of lists of integers.
              - Returns `[]` if no matching <sol> block is found.
              - Returns `[]` if the <sol> block is found but contains no matching <knapsack_i> tags.
              - For each <knapsack_i> tag (including self-closing):
                - Returns `[]` for an empty or whitespace-only assignment (e.g., <knapsack_0></knapsack_0>, <knapsack_1/>).
                - Returns `[int1, int2, ...]` for valid comma-separated integers.
              - The order of lists in the output corresponds to the order of appearance
                of the <knapsack_i> tags in the input text, not necessarily their numerical index.
    """
    sol_pattern = re.compile(r'<[Ss][Oo][Ll][^>]*>\s*(.*?)\s*<\/[Ss][Oo][Ll][^>]*>', re.DOTALL)
    sol_match = sol_pattern.search(text)

    if not sol_match:
        return []

    content = sol_match.group(1)

    # REVISED Pattern for <knapsack_i> tags:
    # Key Change: Use `[^>\n]*` instead of `[^>]*` for attributes part.
    # This prevents the attribute matching from crossing newlines.
    # This ensures `findall` gets distinct <knapsack_X> matches.
    # We still use `re.DOTALL` for the overall `findall` to allow `(.*?)` content to span lines.
    
    knapsack_pattern = re.compile(
        # Standard opening tag with optional attributes (no newlines in attributes)
        r'<[Kk][Nn][Aa][Pp][Ss][Aa][Cc][Kk]_\d+[^>\n]*>\s*'
        r'(.*?)' # Content captured (can span newlines because re.DOTALL is active for findall)
        r'\s*<\/[Kk][Nn][Aa][Pp][Ss][Aa][Cc][Kk]_\d+[^>\n]*>' # Closing tag
        r'|' # OR
        # Self-closing tag with optional attributes (no newlines in attributes)
        r'<[Kk][Nn][Aa][Pp][Ss][Aa][Cc][Kk]_\d+[^>\n]*\/>()' # Empty group for consistency
    , re.DOTALL) # re.DOTALL applies to the whole pattern, allowing (.*?) to match newlines

    knapsack_matches = knapsack_pattern.findall(content)

    if not knapsack_matches:
        return []

    results = []
    for match_tuple in knapsack_matches:
        # Pick the non-empty string from the tuple.
        knapsack_str = match_tuple[0] if match_tuple[0] else match_tuple[1]

        clean_knapsack_str = knapsack_str.strip()

        if not clean_knapsack_str:
            results.append([]) # Empty or whitespace-only, or self-closing
        else:
            try:
                int_list = [int(x.strip()) for x in clean_knapsack_str.split(',') if x.strip()]
                results.append(int_list)
            except ValueError:
                print(f"Warning: Could not convert '{knapsack_str}' to a list of integers. Appending empty list.")
                results.append([])

    return results


def extract_pack_from_string(text):
    """
    Robustly extracts lists of integers from <knapsack_i> tags within <sol>...</sol> block.
    The tag name varies by row index (e.g., <knapsack_0>, <knapsack_1>, etc.).
    Handles various capitalization, potential attributes in tags, common typos,
    and self-closing tags like <knapsack_0/>.
    Corrects issue with `re.findall` consuming too much with `DOTALL` for inner matches.

    Args:
        text (str): The input string to search.

    Returns:
        list: A list of lists of integers.
              - Returns `[]` if no matching <sol> block is found.
              - Returns `[]` if the <sol> block is found but contains no matching <knapsack_i> tags.
              - For each <knapsack_i> tag (including self-closing):
                - Returns `[]` for an empty or whitespace-only assignment (e.g., <knapsack_0></knapsack_0>, <knapsack_1/>).
                - Returns `[int1, int2, ...]` for valid comma-separated integers.
              - The order of lists in the output corresponds to the order of appearance
                of the <knapsack_i> tags in the input text, not necessarily their numerical index.
    """
    sol_pattern = re.compile(r'<[Ss][Oo][Ll][^>]*>\s*(.*?)\s*<\/[Ss][Oo][Ll][^>]*>', re.DOTALL)
    sol_match = sol_pattern.search(text)

    if not sol_match:
        return []

    content = sol_match.group(1)

    # REVISED Pattern for <knapsack_i> tags:
    # Key Change: Use `[^>\n]*` instead of `[^>]*` for attributes part.
    # This prevents the attribute matching from crossing newlines.
    # This ensures `findall` gets distinct <knapsack_X> matches.
    # We still use `re.DOTALL` for the overall `findall` to allow `(.*?)` content to span lines.
    
    knapsack_pattern = re.compile(
        # Standard opening tag with optional attributes (no newlines in attributes)
        r'<[Bb][Ii][Nn]_\d+[^>\n]*>\s*'
        r'(.*?)' # Content captured (can span newlines because re.DOTALL is active for findall)
        r'\s*<\/[Bb][Ii][Nn]_\d+[^>\n]*>' # Closing tag
        r'|' # OR
        # Self-closing tag with optional attributes (no newlines in attributes)
        r'<[Bb][Ii][Nn]_\d+[^>\n]*\/>()' # Empty group for consistency
    , re.DOTALL) # re.DOTALL applies to the whole pattern, allowing (.*?) to match newlines

    knapsack_matches = knapsack_pattern.findall(content)

    if not knapsack_matches:
        return []

    results = []
    for match_tuple in knapsack_matches:
        # Pick the non-empty string from the tuple.
        knapsack_str = match_tuple[0] if match_tuple[0] else match_tuple[1]

        clean_knapsack_str = knapsack_str.strip()

        if not clean_knapsack_str:
            results.append([]) # Empty or whitespace-only, or self-closing
        else:
            try:
                int_list = [int(x.strip()) for x in clean_knapsack_str.split(',') if x.strip()]
                results.append(int_list)
            except ValueError:
                print(f"Warning: Could not convert '{knapsack_str}' to a list of integers. Appending empty list.")
                results.append([])

    return results



def extract_route_from_string_old(answer):
    # Regular expression pattern to find content between <route> and </route>
    pattern = r'<route>(?=[\d, ]*\d)([\d, ]+)</route>'

    # Find all matches
    matches = re.findall(pattern, answer)

    result = []
    for match in matches:
        digits = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
        result.append(digits)

    return result


def extract_group_from_string(answer):
    # Regular expression pattern to find content between <group> and </group>
    pattern = r'<cluster>(?=[\d, ]*\d)([\d, ]+)</cluster>'

    # Find all matches
    matches = re.findall(pattern, answer)

    result = []
    for match in matches:
        digits = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
        result.append(digits)
    
    if len(result) == 0:
        return result
    return result[-1]


def extract_partition_from_string(text):
    # Match start and end tags like <Route> ... </route> (case-insensitive)
    
    route_blocks = re.findall(
        r"<(?i:sub)>\s*(.*?)\s*</(?i:sub)>",
        text,
        flags=re.DOTALL
    )

    result = set([])
    for block in route_blocks:
        # Extract all integers from the block, ignoring delimiters/wrappers
        digits = set(map(int, re.findall(r"\d+", block)))
        result.update(digits)

    return list(result)


def extract_partition_from_string_old(answer):
    # Regular expression pattern to find content between <part> and </part>
    pattern = r'<sub>(?=[\d,\- ]*\d)([\d,\- ]+)</sub>'

    # Find all matches
    matches = re.findall(pattern, answer)

    result = set([])
    for match in matches:
        digits = set(int(num) for num in re.findall(r'-?\d+', match))
        result.update(digits)

    return list(result)


def is_feasible_metaroute(routes, links):
    for s, e in links:
        found_in = 0
        for route in routes:
            if s in route[:-1] and e in route[:-1]:
                found_in += 1
                idx_s = route.index(s)
                idx_e = route.index(e)
                if abs(idx_s - idx_e) != 1 and abs(idx_s - idx_e) != len(route) - 2:
                    return False
        if found_in != 1:
            return False  # Should appear in exactly one route
    return True


def is_feasible_solution(problem: dict):
    problem_type = problem['problem_type']
    sol = problem['sol']
    if problem_type == 'tsp':
        if type(sol[0]) == list:
            sol = sol[0]
        return is_feasible_tsp(sol, sol[0], sol[-1], len(problem['x']) + 1, link=problem.get('link', None))
    if problem_type == 'cvrp':
        return is_feasible_cvrp(sol, depot=problem['depot'], demand=problem['demands'], capacity=problem['capacity'])
    if problem_type == 'knapsack':
        return is_feasible_kp(sol, problem['weights'], problem['capacity'])
    if problem_type == 'bin_pack':
        return is_feasible_bp(sol, problem['items'], problem['capacity'])


def is_feasible_bp(sol, items, capacity):
    # index check
    if max(chain(*sol)) != len(items) - 1 or min(chain(*sol)) != 0:
        return False, "Invalid index"
    if len(set(chain(*sol))) != len(items):
        return False, "Missing packing some items"

    for part in sol:
        if sum(items[i] for i in part) > capacity:
            return False, "Exceeds capacity"

    return True, ""


def is_feasible_tsp(trace, start, end, num, link=None):
    if len(trace) == 0:
        return False, "Empty trace!"
    if trace[0] != start:
        return False, f"Start point should be {start} instead of {trace[0]}"
    if trace[-1] != end:
        return False, f"End point should be {end} instead of {trace[-1]}"

    if link is not None:
        for s, e in link:
            if s not in trace:
                return False, f"Missing point ({s})."
            if e not in trace:
                return False, f"Missing point ({e})."
            sid = trace[:-1].index(s)
            eid = trace[:-1].index(e)
            if abs(sid - eid) != 1 and abs(sid - eid) != len(trace) - 2:
                return False, f"Other points between fixed path ({s},{e})."

    if start == end:
        if len(trace) == num and max(trace) == num-2 and min(trace) == 0:
            return True, None
        return False, "Wrong nodes or missing nodes"

    pc = Counter(trace)

    if len(trace) == len(pc) == num and max(pc.keys()) == max(trace) and min(pc.keys()) == min(trace):
        return True, None

    nice_points = set(range(num))
    miss_points = list(nice_points - set(pc.keys()))
    extra_points = list(set(pc.keys()) - nice_points)
    repeated_points = [p for p in nice_points if pc[p] > 1]

    msgs = []
    if len(miss_points) > 0:
        msgs.append(f"Missing points: {miss_points}")
    if len(extra_points) > 0:
        msgs.append(f"Wrong points: {extra_points}")
    if len(repeated_points) > 0:
        msgs.append(f"Repeated points: {repeated_points}")
    return False, ', '.join(msgs)


def is_feasible_cvrp(routes, depot, demand, capacity, link=None):
    visited = [False] * len(demand)
    for route in routes:
        if len(route) == 0 or route[0] != depot or route[-1] != depot:
            return False, "Each route must start and end at the depot."
        if len(route) == 1:
            continue
        if max(route) >= len(demand) or min(route) < 0:
            return False, "Route contains invalid points."
        total_demand = sum(demand[point] for point in route)
        if total_demand > capacity:
            return False, "Total demand exceeds vehicle capacity."
        for point in route[1:-1]:
            if visited[point]:
                return False, f"Point {point} is visited multiple times."
            visited[point] = True
    
    if not all(visited[1:]):  # Skip depot
        return False, "Not all points are visited."

    return True, None


def is_feasible_kp(selected_items, weights, capacities):
    if len(selected_items) == []:
        return True, ""
    if sum(len(items) for items in selected_items) == 0:
        return True, ""
    if max(chain(*selected_items)) >= len(weights) or min(chain(*selected_items)) < 0:
        return False, "Invalid items in the knapsack."

    if len(set(chain(*selected_items))) != sum(len(items) for items in selected_items):
        return False, "Duplicate items in the knapsack."

    for k_idx, assigned_items in enumerate(selected_items):
        current_knapsack_weight = sum(weights[i] for i in assigned_items)
        if current_knapsack_weight > capacities[k_idx]:
            return False, f"Knapsack {k_idx} capacity exceeded."
    return True, ""


def is_feasible_cluster(cluster_id, num_points, num_groups):
    if len(cluster_id) != num_points:
        return False
    if len(set(cluster_id)) != num_groups:
        return False
    return True


def is_feasible_partition(trace, partitions):
    trace_recovery = []
    for part in partitions:
        trace_recovery.extend(part)
    
    return trace == trace_recovery


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def group_consecutive_labels(labels, points, is_cycle=False):
    if len(labels) == 0:
        return labels

    groups = []
    current_group = [points[0]]

    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            current_group.append(points[i])
        else:
            groups.append(current_group)
            current_group = [points[i]]
    
    # Append the last group
    groups.append(current_group)

    if is_cycle:
        # Merge the first and last groups if their labels are the same
        if groups and len(groups) > 1 and labels[0] == labels[-1]:
            # groups[-1].extend(groups[0])
            head = groups.pop(0)
            groups[-1].extend(head)

    return groups


def get_zones(solution, active_elements, problem_type = 'tsp'):
    """
    Extracts zones from a solution based on active elements.
    
    Parameters:
    - solution: list of list of elements
    - active_elements: A list of active elements to consider.
    - is_routing: If True, the solution is treated as a routing problem.
    
    Returns:
     Active zones and Static zones, with meta route if routing problem
    """
    if problem_type == 'assignment':
        active_zones = [[ele] for ele in active_elements]
        static_zones = [[ele for ele in chain(*solution) if ele not in active_elements]]
        return active_zones, static_zones, []

    if problem_type == 'knapsack':
        active_zones = [[ele] for ele in active_elements]
        static_zones = [[ele for ele in sol if ele not in active_elements] for sol in solution]
        return active_zones, static_zones, []

    if problem_type == 'tsp':
        route = solution[0]
        meta_route = []
        start_idx = route.index(active_elements[0])
        zones, part = [], []
        for node in route[start_idx:-1] + route[:start_idx]:
            if node not in active_elements:
                part.append(node)
                continue
            # For active element, add the cuurent node, add part to zone, add part to meta
            zones.append([node])
            if len(part) == 0:
                meta_route.append(node)
                continue
            meta_route.append(part[0])
            if part[-1] != part[0]:
                meta_route.append(part[-1])
            zones.append(part)
            meta_route.append(node)
            
            part = []

        if len(part) > 0:
            zones.append(part)
            meta_route.append(part[0])
            if part[-1] != part[0]:
                meta_route.append(part[-1])

        active_zones = [zone for zone in zones if len(zone) == 1]
        static_zones = [zone for zone in zones if len(zone) > 1]
        return active_zones, static_zones, meta_route

    # CVRP
    depot = solution[0][0]
    zones = []
    meta_routes = []

    for route in solution:
        part = []
        meta_route = []
        for node in route:
            if node not in active_elements:
                part.append(node)
                continue
            # For active element, add to zone, add part to meta
            if node != depot:
                zones.append([node])
            if len(part) == 0:
                meta_route.append(node)
                continue
            if part == [depot]:
                part = []
                meta_route.append(depot)
                meta_route.append(node)
                continue
            if part[-1] != part[0]:
                meta_route.append(part[0])
                meta_route.append(part[-1])
            if len(part) == 1:
                meta_route.append(part[0])
            meta_route.append(node)
            zones.append(part)
            part = []

        if len(part) == 0:
            if len(meta_route) > 0:
                meta_routes.append(meta_route)
            continue
        if part[-1] != part[0]:
            meta_route.append(part[0])
            meta_route.append(part[-1])
        elif part != route:
            meta_route.append(part[0])
        if part != [depot]:
            zones.append(part)
        if len(meta_route) > 0:
            meta_routes.append(meta_route)

    active_zones = [zone for zone in zones if len(zone) == 1]
    static_zones = [zone for zone in zones if len(zone) > 1]
    if depot in active_elements:
        active_zones.append([depot])

    return active_zones, static_zones, meta_routes


def get_segments_from_route(route, destory_points):
    '''
    meta_rote should start with a destory point
    '''
    result = []
    start_idx = route.index(destory_points[0])
    part = []
    meta_route = []
    for node in route[start_idx:-1] + route[:start_idx]:
        if node not in destory_points:
            part.append(node)
            continue
    
        if len(part) == 0:
            result.append([node])
            meta_route.append(node)
            continue

        result.append(part)
        result.append([node])

        if len(part) == 1:
            meta_route.append(part[0])
        else:
            meta_route += [part[0], part[-1]]
        meta_route.append(node)
        part = []
    
    if len(part) == 0:
        meta_route.append(meta_route[0])
        return result, meta_route

    result.append(part)
    if len(part) == 1:
        meta_route.append(part[0])
    else:
        meta_route += [part[0], part[-1]]

    meta_route.append(meta_route[0])
    return result, meta_route


def recover_routes(solution, meta_routes):
    results = [route for route in meta_routes if len(route) > 1 and route[0] == route[-1]]
    part_routes = [route for route in meta_routes if route not in results]
    for sol in solution:
        valid_routes = []
        for route in part_routes:
            if route[0] in sol and route[-1] in sol:
                valid_routes.append(route)
        result = recover_route(sol, valid_routes)
        if result[-1] != sol[-1]:
            result.append(sol[-1])
        results.append(result)
    return results


def recover_route(sol, parts):
    if len(parts) == 0:
        return sol
    short_map = {}
    for segment in parts:
        if len(segment) == 1:
            continue
        short_map[(segment[0],segment[-1])] = [p for p in segment][1:]
        short_map[(segment[-1],segment[0])] = [p for p in segment[::-1]][1:]
    
    result = [sol[0]]
    used_link = set([])
    for s,e in zip(sol, sol[1:]):
        if (s,e) in used_link:
            result.append(e)
        elif (s,e) in short_map:
            result += short_map[(s,e)]
            used_link.add((s,e))
            used_link.add((e,s))
        else:
            result.append(e)

    return result


def gather_answers(current_epoch=0, reject_count=0, max_reflection=3, root='results'):
    answers = []

    for ep in range(current_epoch - reject_count, current_epoch):
        possible_files = [f'{root}/part_answer_{ep}_r{nr}.txt' for nr in range(max_reflection, 0, -1)] + [f'{root}/part_answer_{ep}.txt']
        for file in possible_files:
            if os.path.isfile(file):
                with open(file, 'r') as f:
                    answer = f.read()
                answers.append(answer)
                break

    return answers


if __name__ == '__main__':
    # sol = [0,28,2,0]
    # parts = [[28,0],[0,12,7,2]]
    # print(recover_route(sol, parts))
    # solution = [[0,1,2,3,0], [0,4,5,6,0]]
    # active_ele = [0,3]
    # print(get_zones(solution, active_ele, 'cvrp'))

    with open ('results/destory_llm+repair_llm/knapsack/gpt-4.1_0/mkp_393_5_0/dialogs/revise_answer_0_r0.txt', 'r') as f:
        text = f.read()
        print(extract_assignment_from_string(text))
def gen_partition_prompt(solution, problem_type, reviser_size=10, previous_answers=[]):
    """
    Generate a general prompt for partitioning nodes/items in an optimization solution.

    Args:
        - solution: list of lists of indices representing the current solution
        - problem_type: one of 'tsp', 'cvrp', 'assignment', 'knapsack', 'bin_pack', etc.
        - reviser_size: Max number of indices to propose for improvement.
        - previous_answers (list): List of stringified previous cluster suggestions.
    Returns:
        str: The prompt to be sent to the model.
    """
    label = {
        "tsp": "nodes",
        "cvrp": "nodes",
        "knapsack": "items",
        "assignment": "assignments",
        "bin_pack": "items"
    }.get(problem_type, "elements")

    meta_prompt = ""
    meta_prompt += f"For the given optimization problem ({problem_type if problem_type != 'knapsack' else 'multiple knapsack'}), my current solution is:\n\n"
    meta_prompt += "```\n"
    if len(solution) == 1:
        meta_prompt += ",".join(str(i) for i in solution[0]) + "\n"
    else:
        for idx,sol in enumerate(solution):
            meta_prompt += f"#{idx+1}: " +  ",".join(str(i) for i in sol) + "\n"
    meta_prompt += "```\n\n"

    if problem_type == "tsp":
        meta_prompt += "The route is not the shortest. I want to improve it by selecting a subset of nodes, removing their edges, and reconnecting them optimally.\n"
    elif problem_type == "cvrp":
        meta_prompt += "These vehicle routes are not optimal. I want to improve them by selecting some customer nodes from the current solution, removing their edges, and reconnecting them optimally.\n"
    elif problem_type == "assignment":
        meta_prompt += "The current assignments might not be optimal. I want to refine them by identifying a small subset of assignments to reconsider.\n"
    elif problem_type == "knapsack":
        meta_prompt += "The current selection of items might not be optimal. I want to identify a subset of items to reconsider for packing.\n"
    elif problem_type == "bin_pack":
        meta_prompt += "The current packing might not be space-efficient. I want to find a group of items that could be repacked more optimally.\n"
    else:
        meta_prompt += "I want to improve this solution by selecting a meaningful subset of elements that may be rearranged, reassigned, or recomputed.\n"

    meta_prompt += f"Please identify no more than {reviser_size} {label} that could be improved. These {label} should form a meaningful cluster or group.\n\n"

    if problem_type == 'knapsack':
        meta_prompt += "The items should be at least 1 from selected items, and at least 1 from unselected items, to ensure a meaningful partition.\n\n"

    if previous_answers:
        meta_prompt += "\nBelow are some answers that my repair agent could not improve:\n"
        meta_prompt += "\n" + "\n\n".join(f"**Answer #{i}**:\n'''\n{answer}\n'''" for i, answer in enumerate(previous_answers)) + "\n"
        meta_prompt += "\nAnalyze these examples and suggest a better subset.\n"

    return meta_prompt


def gen_revise_prompt(data, n_examples=None, num_decimals=3):
    """
    Generate the system and user prompts for the LLM reviser.

    Args:
        - data (dict): sub-problem instance; must contain 'problem_type' plus the
          problem-specific fields ('locations', 'start'/'end' or 'depot', 'link', ...).
        - n_examples (list): (error message, solution string) pairs of infeasible answers.
        - num_decimals: kept for API compatibility, currently unused.
    Returns:
        (str, str): system prompt and user prompt.
    """
    sys_prompt = """
Act as an expert in combinatorial optimization. Your goal is find the best solution with given constraints, use any heuristic as you can.
**Goal: Find the best solution, avoid the infeasible solutions.**
**Requirements:**
- Check the feasibility of your solution to ensure the above conditions.
"""
    problem_type = data['problem_type']
    if problem_type == 'cvrp':
        sys_prompt += """
- Must visit all the points exactly once, except the depot.
- The start, end or depot points must remain fixed.
- The fixed path prevent any other visits in between, allow reversing of the fixed path where necessary.

"""
        sys_prompt += "OUTPUT FORMAT: After your solving, present your answer as:\n\n<sol>\n<route>0,1,2,...,0</route>\n<route>0,3,4,...,0</route>\n...\n</sol>\n\n"

    if problem_type == 'tsp':
        sys_prompt += """
- Must visit all the points exactly once, except the depot.
- The start, end or depot points must remain fixed.
- The fixed path prevent any other visits in between, allow reversing of the fixed path where necessary.

"""
        sys_prompt += "OUTPUT FORMAT: After your solving, present your answer as:\n\n<route>0,1,2,...,0</route>\n\n"
        locations = data['locations']
        start, end = data['start'], data['end']
        link = data['link']

    if problem_type == 'cvrp':
        locations = data['locations']
        start = data['depot']
        link = data['link']
        capacity = data['capacity']
        demands = data['demands']
        meta_sol = data['metasol']
    if problem_type == 'knapsack':
        sys_prompt += """
- Selected items weights must be less than or equal to the knapsack's capacity.

"""
        sys_prompt += "OUTPUT FORMAT: After your solving, present your answer as:\n\n<sol>\n<knapsack_0>0,1,2,...</knapsack_0>\n<knapsack_1>3,4,...</knapsack_1>\n...\n</sol>\nWhere knapsack_i is the i-th knapsack\n"

        weights = data['weights']
        values = data['values']
        capacity = data['capacity']

        meta_prompt = "Problem:\nGiven a set of items with their values and weights, your task is to find the best packing solution that maximizes the total value while respecting the each knapsack's capacity.\n\n"
        meta_prompt += f"""
**Items' weights**: {weights}

**Items' values**: {values}

**Knapsack's capacities**: {capacity}

"""
    if problem_type == 'bin_pack':
        sys_prompt += """
- Packed items weights must be less than or equal to the bin's capacity.

"""
        sys_prompt += "OUTPUT FORMAT: After your solving, present your answer as:\n\n<sol>\n<bin_0>0,1,2,...</bin_0>\n<bin_1>3,4,...</bin_1>\n...\n</sol>\nWhere bin_i is the i-th bin, the nested value are the item indices only, each item should be packed only once.\n"

        items = data['items']
        capacity = data['capacity']

        meta_prompt = "Problem:\nGiven a set of items with their weights, your task is to find the best packing solution that maximizes the total value while respecting the each bin's capacity.\n\n"
        meta_prompt += f"""
**Items**: {items}

**Bin's capacity**: {capacity}

"""
    if problem_type in ['tsp', 'cvrp']:
        points_str = "```csv\nx,y\n"
        points_str += "\n".join(f"{coord[0]},{coord[1]}" for coord in locations)
        points_str += "\n```\n"
        meta_prompt = f"Given a set of points must to visit, your task is to find the **shortest total distance** route(s) while respecting the following constraints.\n\n"
        meta_prompt += f"**Input of points (index starts from 0):**\n"
        meta_prompt += points_str + "\n"
        if problem_type == 'tsp':
            meta_prompt += f"**Start point index**: {start}\n"
            meta_prompt += f"**End point index**: {end}\n\n"

            meta_prompt += "Init solution:\n"
            sol_str = "  <route>" + ",".join(str(i) for i in list(range(len(locations))) + [0]) + "</route>"
            meta_prompt += "<sol>\n" + sol_str + "\n</sol>\n\n"
        else:
            meta_prompt += f"**Depot point index**: {start}\n"
            meta_prompt += f"**Vehicle Capacity**: {capacity}\n"
            meta_prompt += f"**Demand of each point**: {demands}\n"
            meta_prompt += f"You can use multiple vehicles, and each vehicle go with one route, try any heuristics, you can start by analysing the initial solution below and improve it.\n\n"

            meta_prompt += "Init solution:\n"
            sol_str = "\n".join(
                "  <route>" + ",".join(str(i) for i in sol) + "</route>" for sol in meta_sol
            )
            meta_prompt += "<sol>\n" + sol_str + "\n</sol>\n\n"

        if link is not None:
            meta_prompt += "**Fixed visiting path**:\n"
            meta_prompt += "```\n" + "\n".join(
                f"({s},{e})" for s, e in link
            ) + "\n```\n\n"

    if n_examples:
        meta_prompt += "\n\n**Negative Examples:**\n\n"
        for err_msg, trace in n_examples:
            meta_prompt += trace.strip() + f", Reason: {err_msg}\n\n"

    return sys_prompt, meta_prompt

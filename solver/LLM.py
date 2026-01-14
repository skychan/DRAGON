from heapq import heapify, heappush, heappop, heappushpop
from baselines.metrics import Metrics
from utils import extract_partition_from_string, extract_route_from_string, extract_assignment_from_string, extract_pack_from_string, evaluate_objective, is_feasible_solution, gather_answers, get_zones
from openai import OpenAI
from generate_prompts import gen_revise_prompt, gen_partition_prompt
from dotenv import load_dotenv
import json
import os
import time
import numpy as np
from scipy.spatial import distance_matrix
from itertools import chain

# np.random.seed(42)

load_dotenv(override=True)


def llm_solver(data, num_decimals = 0, save_root=None, epoch=0, revise_num = 0, metric : Metrics = None, n_examples = None):
    '''
    point_coord: include the start and the end points
    '''
    # if len(points) <= 3:
    #     return points

    # if p_examples is None:
    #     # random_route = [i for i in range(start+1, end)]
    #     # heu_route = list(range(start, end+1))
    #     heu_route = points + [points[0]]
    #     heu_dist = evaluate_distance(X, Y, heu_route, num_decimals)
    #     p_examples = []

    #     p_examples.append(
    #         (-heu_dist,
    #         "<route>" + ",".join(str(i) for i in heu_route) + f"</route>")
    #         )
    #     heapify(p_examples)

    # point_coord = {i: (X[i], Y[i]) for i in points}

    print(f"======= Revise number: {revise_num} =========")

    problem_type = data['problem_type']
    dialog_root = os.path.join(save_root, 'dialogs')
    problem_root = save_root
    # start, end = data['start'], data.get('end', None)

    # link = [(p_map_rev[zone[0]], p_map_rev[zone[-1]]) for zone in data['static_zone']]
    # link = data['link']
    # locations = data['locations']
    # X, Y = zip(*locations)

    system_prompt, user_prompt = gen_revise_prompt(data, num_decimals=num_decimals, n_examples=n_examples)

    with open(os.path.join(dialog_root, f"revise_prompt_{epoch}_r{revise_num}.txt"), "w") as f:
        f.write(user_prompt)

    if os.path.isfile(os.path.join(dialog_root, f"revise_answer_{epoch}_r{revise_num}.txt")):
        with open(os.path.join(dialog_root, f"revise_answer_{epoch}_r{revise_num}.txt"), "r") as f:
            answer = f.read()
    else:
        if metric.use_ci:
            answer = get_llm_response_ci(save_root, user_prompt, sys_prompt=system_prompt, mode = 'revise', metric=metric, problem_type=problem_type)
        else:
            answer = get_llm_response(system_prompt, user_prompt, metric)
        with open(os.path.join(dialog_root, f"revise_answer_{epoch}_r{revise_num}.txt"), "w") as f:
            f.write(answer)
    # with open(os.path.join(save_root, f"revise_answer_{epoch}_r1.txt"), "r") as f:
    #     answer = f.read()
    if problem_type == 'knapsack':
        # extract the route from the answer
        assignment = extract_assignment_from_string(answer)
    elif problem_type == 'bin_pack':
        packing = extract_pack_from_string(answer)
    else:
        traces = [trace for trace in extract_route_from_string(answer) if len(trace) > 0]
    # end, start = link[-1]
    # problem_data = {
    #     'problem_type': problem_type,
    #     'start': start,
    #     'end': end,
    #     'x': X,
    #     'y': Y,
    #     'locations': locations,
    #     'link': link
    # }
    problem_data = {k:v for k,v in data.items()}

    have_feasible_solution = False

    solutions = []
    if problem_type == 'cvrp':
        clean_trace = []
        for trace in traces:
            clean_trace.append([0] + [t for t in trace if t != 0] + [0])
        problem_data['sol'] = clean_trace
        is_feasible, error_msg = is_feasible_solution(problem_data)
        if not is_feasible:
            if n_examples is None:
                n_examples = []
            n_examples.append((error_msg, "<sol>\n" + "\n".join(
                "<route>" + ",".join(str(i) for i in trace) + "</route>" for trace in clean_trace
            ) + "\n</sol>\n"))
        else:
            dist = evaluate_objective(problem_data, num_decimals)
            solution = (dist, "\n".join(
                "<route>" + ",".join(str(i) for i in trace) + "</route>" for trace in clean_trace
            ))
            solutions.append(solution)
            have_feasible_solution = True
    elif problem_type == 'tsp':
        for trace in traces:
            if trace[-1] != 0:
                trace.append(0)
            problem_data['sol'] = [trace]
            is_feasible, error_msg = is_feasible_solution(problem_data)
            if not is_feasible:
                if n_examples is None:
                    n_examples = []
                n_examples.append((error_msg, "<route>" + ",".join(str(i) for i in trace) + f"</route>"))
            else:
                dist = evaluate_objective(problem_data, num_decimals)
                solution = (dist, "<route>" + ",".join(str(i) for i in trace) + f"</route>")
                # if len(p_examples) <= 10:
                #     heappush(p_examples, solution)
                # else:
                #     heappushpop(p_examples, solution)
                solutions.append(solution)
                have_feasible_solution = True
    elif problem_type == 'knapsack':
        problem_data['sol'] = assignment
        is_feasible, error_msg = is_feasible_solution(problem_data)
        if not is_feasible:
            if n_examples is None:
                n_examples = []
            n_examples.append((error_msg, "<sol>\n" + "\n".join(
                f"<knapsack_{k}>" + ",".join(str(i) for i in assign) + f"</knapsack_{k}>" for k, assign in enumerate(assignment)
            ) + "\n</sol>\n"))
        else:
            obj_val = evaluate_objective(problem_data, num_decimals)
            solution = (-obj_val, "<sol>\n" + "\n".join(
                f"<knapsack_{k}>" + ",".join(str(i) for i in assign) + f"</knapsack_{k}>" for k, assign in enumerate(assignment)
            ) + "\n</sol>\n")
            solutions.append(solution)
            have_feasible_solution = True
    elif problem_type == 'bin_pack':
        problem_data['sol'] = packing
        is_feasible, error_msg = is_feasible_solution(problem_data)
        if not is_feasible:
            if n_examples is None:
                n_examples = []
            n_examples.append((error_msg, "<sol>\n" + "\n".join(
                f"<bin_{k}>" + ",".join(str(i) for i in pack) + f"</bin_{k}>" for k, pack in enumerate(packing)
            ) + "\n</sol>\n"))
        else:
            obj_val = evaluate_objective(problem_data, num_decimals)
            solution = (obj_val, "<sol>\n" + "\n".join(
                f"<bin_{k}>" + ",".join(str(i) for i in pack) + f"</bin_{k}>" for k, pack in enumerate(packing)
            ) + "\n</sol>\n")
            solutions.append(solution)
            have_feasible_solution = True

    if have_feasible_solution:
        best_str = sorted(solutions)[0][1]
        if problem_type == 'knapsack':
            result = extract_assignment_from_string(best_str)
        elif problem_type == 'bin_pack':
            result = extract_pack_from_string(best_str)
        else:
            result = extract_route_from_string(best_str)
        if problem_type in ['cvrp', 'knapsack', 'bin_pack']:
            return result
        return [result[0]]

    if revise_num < 3:
        return llm_solver(data, num_decimals = num_decimals, save_root=problem_root, epoch=epoch, revise_num = revise_num+1, metric=metric, n_examples=n_examples)

    return []
    # best_str = sorted(p_examples, reverse=True)[0][1]
    # result = extract_route_from_string(best_str)
    # return result[0]


def Initialize(points, x, y, start, end, num_decimals = 0):
    '''
    Points are exclude start and end
    '''
    point_coord, p_map, p_map_rev = point_reindex(points, x, y, start, end)

    # # prepare for ortools
    # locations = [point_coord[i] for i in range(len(point_coord))]
    # X, Y = zip(*locations)
    # ortools_route = orsolver(locations, X, Y, p_map_rev[start], p_map_rev[end])

    random_traces = []
    for _ in range(5):
        random_trace = points[:]
        np.random.shuffle(random_trace)
        random_trace = [start] + random_trace + [end]
        dist = evaluate_distance(x, y, random_trace, num_decimals)
        random_trace_str = "<trace>" + ",".join(str(p) for p in random_trace) + "</trace>"
        random_traces.append((-dist, random_trace_str))
    
    # Solve directly
    prompt = gen_revise_prompt(point_coord, p_map_rev[start], p_map_rev[end], random_traces)

    result = llm_solver(prompt, [], None, 0,  p_map_rev[start], p_map_rev[end], point_coord, best_threshold=1)
    # Replace with LLM
    # result = np.random.permutation(range(len(points)))

    trace = [p_map[i] for i in result]

    return trace


def get_llm_response(sys_prompt, user_prompt, metric: Metrics = None):
    if metric.model_name == 'deepseek-reasoner':
        DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    else:
        client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model=metric.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=1.2
        )
        answer = completion.choices[0].message.content

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        metric.add_input_token_len(usage.prompt_tokens)
        metric.add_output_token_len(usage.completion_tokens)
        metric.increment_api_calls()

        return answer
    except Exception as e:
      print(f"An error occurred: {e}")


def get_llm_response_ci(root_path, user_prompt, sys_prompt='', mode='divide', metric : Metrics = None, problem_type='tsp'):

    # if config exists, use it
    client = OpenAI()

    if mode == 'divide':
        if os.path.exists(os.path.join(root_path, 'config.json')):
            with open(os.path.join(root_path, 'config.json')) as f:
                config = json.load(f)
            assist_id = config['assistant_id']
            file_id = config['file_id']
        else:

            # Upload a file with an "assistants" purpose
            file = client.files.create(
                file=open(os.path.join(root_path, "problem.json"), "rb"),
                purpose='assistants'
            )
            if problem_type == 'tsp':
                assist_instruct = """
You are an expert in optimization with smart heuristics. The user will provide you an initial routing solution, whose node coordination values are attached as a `problem.json` file, formatted as ```{x: [x1, x2, ...], y: [y1, y2, ...]}```. Help the user analysis the initial solution and point out which part is yet to be optimized using any creative heuristic methods you can. Ensure to output the answer in the specified required format.

**Format**: Return only the node index, enclosed in <sub> and </sub>, separated by commas.
For example: ```<sub>1,2,3</sub>```

"""

            elif problem_type == 'cvrp':
                assist_instruct = """
You are an expert in optimization with smart heuristics. The user will provide you an initial routing solution, whose node coordination values are attached as a `problem.json` file, formatted as ```{x: [x1, x2, ...], y: [y1, y2, ...], demands: [d1, d2, ...], capacity: c, depot: n0}```. Help the user analysis the initial solution and point out which part is yet to be optimized using any creative heuristic methods you can. Ensure to output the answer in the specified required format.

**Format**: Return only the node index, enclosed in <sub> and </sub>, separated by commas.
For example: ```<sub>1,2,3</sub>```.

"""
            assistant = client.beta.assistants.create(
                name="Heuristic Expert",
                instructions=assist_instruct,
                model=metric.model_name,
                tools=[{"type": "code_interpreter"}],
                tool_resources={
                    "code_interpreter": {
                        "file_ids": [file.id]
                    }
                }
            )
            assist_id = assistant.id
            file_id = file.id
            # save assistant id
            with open(os.path.join(root_path, 'config.json'), 'w') as f:
                json.dump({'assistant_id': assistant.id, 'file_id': file.id}, f)
    elif mode == 'revise':
        # assist_id = 'asst_aFJlyv1AK6tyvAL8qPGcPrfV'
        assistant = client.beta.assistants.create(
                name="Heuristic Expert",
                # instructions="You are an expert in optimization with smart heuristics. The user will provide you a routing solution that travels the nodes list in the given csv file. Help the users solve their problems using any creative heuristic methods you can. Ensure to output the answer in the specified required format.",
                model=metric.model_name,
                tools=[{"type": "code_interpreter"}],
                # tool_resources={
                #     "code_interpreter": {
                #         "file_ids": [file.id]
                #     }
                # }
            )
        assist_id = assistant.id

    if sys_prompt != '':
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "assistant",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        )
    else:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assist_id,
        tool_choice='required'
    )
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(15) # Wait for 1 second
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    out_pred = ""
    if run.status == 'completed':
        usage = run.usage
        metric.add_input_token_len(usage.prompt_tokens)
        metric.add_output_token_len(usage.completion_tokens)
        metric.increment_api_calls()
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        try:
            out_pred = messages.data[0].content[0].text.value
        except:
            pass

    return out_pred


def L_partition_trace(trace, x, y):
    is_loop = False
    if trace[0] == trace[-1]:
        trace_for_part = trace[:-1]
        is_loop = True
    else:
        trace_for_part = trace

    partitions = llm_partitioner(trace_for_part, x, y)

    if not is_loop:
        return partitions

    boundary_trace = partitions[-1] + partitions[0]

    boundary_partitions = llm_partitioner(boundary_trace, x, y)

    boundary_partitions[-1].append(partitions[1][0])
    result = partitions[1:-1] + boundary_partitions

    return result


def worst_removal(solution, problem_type='tsp', reviser_size = 10, problem_root = None):
    """
    Removes the customers whose removal leads to the largest cost savings.

    Parameters:
    - solution: List of nodes representing the current route (e.g., [0, 1, 2, 3, 0])
    - distance_matrix: 2D list or array of distances between nodes
    - rate: Fraction of customers to remove (0 < rate < 1)

    Returns:
    - removed_customers: List of removed customers
    """

    # calc dist_mat
    with open(os.path.join(problem_root, "problem.json"), 'r') as f:
        data = json.load(f)
    
    points = np.column_stack((data['x'], data['y']))
    dist_matrix = distance_matrix(points, points)

    # Compute savings if each customer is removed
    savings = {}
    for i in range(1, len(solution[0]) - 1):
        prev_node = solution[0][i - 1]
        current_node = solution[0][i]
        next_node = solution[0][i + 1]

        # Cost of keeping current_node: prev → current → next
        original_cost = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node]
        # Cost of skipping current_node: prev → next
        new_cost = dist_matrix[prev_node][next_node]
        savings[current_node] = original_cost - new_cost

    # Select top nodes to remove by savings (most benefit first)
    removed_solution = sorted(savings, key=savings.get, reverse=True)[:reviser_size]

    return removed_solution


def llm_partitioner(solution, reviser_size=10, problem_type='tsp', save_root=None, epoch=0, reject_count=0, metric : Metrics = None):
    '''
    Input:
        - solution: list of list of elements
        - locations: list of coordinates
        - reviser_size: size of reviser
    Output:
        - destoryed points: list
    '''
    max_reflection = 3
    dialog_root = os.path.join(save_root, 'dialogs')
    problem_root = save_root

    if os.path.isfile(f'{dialog_root}/part_prompt_{epoch}.txt'):
        with open(f'{dialog_root}/part_prompt_{epoch}.txt', 'r') as f:
            prompt = f.read()
    else:
        previous_answers = gather_answers(current_epoch=epoch, reject_count=reject_count, max_reflection=max_reflection, root=dialog_root)
        prompt = gen_partition_prompt(solution, problem_type, reviser_size, previous_answers)
        if not metric.use_ci:
            if problem_type == 'tsp':
                sys_prompt = """
You are an expert in optimization with smart heuristics. The user will provide you an initial routing solution, whose node coordination values are formatted as ```{x: [x1, x2, ...], y: [y1, y2, ...]}```. Help the user analysis the initial solution and point out which part is yet to be optimized using any creative heuristic methods you can. Ensure to output the answer in the specified required format.

**Format**: Return only the node index, enclosed in <sub> and </sub>, separated by commas.
For example: ```<sub>1,2,3</sub>```

"""
            elif problem_type == 'knapsack':
                sys_prompt = """
You are an expert in optimization with smart heuristics. The user will provide you an initial packing solution, whose data are formatted as ```{capacities: [c1, c2, ...], values: [v1, v2, ...], weights: [w1, w2, ...]}```. Help the user analysis the initial solution and point out which part is yet to be optimized using any creative heuristic methods you can. Ensure to output the answer in the specified required format.

**Format**: Return only the node index, enclosed in <sub> and </sub>, separated by commas.
For example: ```<sub>1,2,3</sub>```
"""
            elif problem_type == 'bin_pack':
                sys_prompt = """
You are an expert in optimization with smart heuristics. The user will provide you an initial packing solution, whose data are formatted as ```{capacity: [c1, c2, ...], num_items: N, items: [i1, i2, ...]}```. Help the user analysis the initial solution and point out which part is yet to be optimized using any creative heuristic methods you can. Ensure to output the answer in the specified required format.

**Format**: Return only the node index, enclosed in <sub> and </sub>, separated by commas.
For example: ```<sub>1,2,3</sub>```
"""

            with open(os.path.join(problem_root, 'problem.json'), 'r') as f:
                problem_str = json.dumps(json.load(f))
            prompt += f"""
The problem inputs are:

```json
{problem_str}
```
"""
        with open(f'{dialog_root}/part_prompt_{epoch}.txt', 'w') as f:
            f.write(prompt)

    # return destoryed_points
    current_reflection_num = 0
    answer = None
    for nr in range(max_reflection-1, 0, -1):
        if os.path.isfile(f'{dialog_root}/part_answer_{epoch}_r{nr}.txt'):
            with open(f'{dialog_root}/part_answer_{epoch}_r{nr}.txt', "r") as f:
                answer = f.read()
            current_reflection_num = nr
            break

    if answer is None and os.path.isfile(f'{dialog_root}/part_answer_{epoch}.txt'):
        with open(f'{dialog_root}/part_answer_{epoch}.txt', 'r') as f:
            answer = f.read()
    else:
        if not metric.use_ci:
            answer = get_llm_response(sys_prompt, prompt, metric)
        else:
            answer = get_llm_response_ci(problem_root, prompt, metric=metric, problem_type=problem_type)
        with open(f'{dialog_root}/part_answer_{epoch}.txt', 'w') as f:
            f.write(answer)
    if problem_type == 'knapsack':
        with open(os.path.join(problem_root, 'problem.json'), 'r') as f:
            data = json.load(f)

        active_elements = [p for p in extract_partition_from_string(answer) if 0 <= p < len(data['values'])]
    else:
        active_elements = [p for p in extract_partition_from_string(answer) if p in chain(*solution)]  # remove unexpected nodes

    while len(active_elements) == 0:
        if current_reflection_num == max_reflection:
            # raise Exception("No valid answer")
            print("No valid answer")
            break

        if current_reflection_num == 0:
            prompt += f"You previous answer is: \n```{answer}```\n"
        else:
            answers = [answer]
            for i in range(current_reflection_num, 0, -1):
                with open(f'{dialog_root}/part_answer_{epoch}_r{i}.txt', 'r') as f:
                    answer = f.read()
                answers.append(answer)
            with open(f'{dialog_root}/part_answer_{epoch}.txt', 'r') as f:
                answer = f.read()
            answers.append(answer)
            prev_trial = '\n\n'.join(f'Trial_{i+1}:\n{ans}' for i, ans in enumerate(answers))
            prompt += f"You previous answers are: \n```{prev_trial}```\n\n"
        prompt += "The number of valid node index in each answer(s) above is 0, please rethink and try again."
        if not metric.use_ci:
            answer = get_llm_response(sys_prompt, prompt, metric)
        else:
            answer = get_llm_response_ci(problem_root, prompt, metric=metric)
        current_reflection_num += 1
        with open(f'{dialog_root}/part_answer_{epoch}_r{current_reflection_num}.txt', 'w') as f:
            f.write(answer)
        active_elements = [p for p in extract_partition_from_string(answer) if p in chain(*solution)]  # remove unexpected nodes

    return active_elements


def random_partition(solution, problem_type='tsp', reviser_size = 10):
    '''
    solution: list of list of elements
    Select points to detach from the given route
        candidates for such problem that solution will not contain all the elements
    '''
    if problem_type != 'knapsack':
        elements = list(set(chain(*solution)))
        num_selected = np.random.randint(1, min(reviser_size // 2 + 1, len(elements)))
        # num_selected = reviser_size
        active_elements = np.random.choice(elements, num_selected, replace=False)

        return active_elements.tolist()

    # sample from both sloution and candidates, add up to reviser_size
    solution_elements, candidate_elements = solution
    num_sample_total = np.random.randint(2, min(reviser_size // 2 + 1, len(solution_elements) + len(candidate_elements)))
    num_sample_solution = np.ceil(num_sample_total * len(solution_elements) / (len(solution_elements) + len(candidate_elements))).astype(int)
    num_sample_candidates = num_sample_total - num_sample_solution
    
    active_elements_solution = np.random.choice(solution_elements, num_sample_solution, replace=False)
    active_elements_candidates = np.random.choice(candidate_elements, num_sample_candidates, replace=False)
    active_elements = active_elements_solution.tolist() + active_elements_candidates.tolist()

    return active_elements


def meta_partition(solution, problem_type = 'tsp', reviser_size = 10, method='random', save_root=None, epoch = 0, reject_count = 0, metric : Metrics = None):
    '''
    Input:
        - solution: list or list of list of elements
    Output:
        - active_zone: the segments of the route, follow the order of the original route
            - meta_points: the segments of the route, follow the order of the original route
            - meta_route: only the link points follow the original route
        - static_zone: the links
            - short_cuts: the links
    '''
    # if solution is not a list of list, convert it to a list of list
    if not isinstance(solution[0], list):
        solution = [solution]

    if method == 'random':
        active_elements = random_partition(solution, problem_type=problem_type, reviser_size=reviser_size)
    elif method == 'llm':
        active_elements = llm_partitioner(solution, reviser_size, problem_type=problem_type, save_root=save_root, epoch=epoch, reject_count=reject_count, metric=metric)
    elif method == 'heuristic':
        active_elements = worst_removal(solution, problem_type=problem_type, problem_root=save_root)
    else:
        raise NotImplementedError

    active_zone, static_zone, meta_sol = get_zones(solution, active_elements, problem_type=problem_type)

    return active_zone, static_zone, meta_sol

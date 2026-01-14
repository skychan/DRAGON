from solver import orsolver, heuristic_solver, llm_solver, meta_partition
import argparse
from utils import read_co_problem_instance, evaluate_objective

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default="instances/tsp200_test.pkl", type=str, help='The input file name')
    parser.add_argument('--time_limit', default=10, type=int, help='The time limit for each task')

    return parser.parse_args()


def solve():
    data = read_co_problem_instance(problem_type='cvrp', file_path=args.input_file, task_id=-1, num_decimals=0)
    data['problem_type'] = 'cvrp'
    X, Y = zip(*data['node_coord'])
    data['demands'] = data['demand'].tolist()
    data['locations'] = data['node_coord']
    data['depot'] = 0
    data['capacity'] = [data['capacity']]
    data['link'] = []
    result = orsolver(data, quick=True, time_limit=args.time_limit)
    data['sol'] = result

    print(f"Total distance = {evaluate_objective(data)}")


if __name__ == '__main__':
    args = parse_args()
    solve()


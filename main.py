from utils import setup_logging
import argparse
from dragon import solve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default="instances/tsp200_test.pkl", type=str, help='The input file name')
    parser.add_argument('-t', '--task', default=-1, type=int, help='The task number, only for random tsp cases')
    parser.add_argument('-r', '--reviser_size', default=20, type=int, help='The reviser size, e.g. [10,20,50,100]')
    parser.add_argument('-n', '--iterations', default=10, type=int, help='The number of iterations')
    # parser.add_argument('-l', '--ues_llm', action='store_true', help='Use LLM or not')
    parser.add_argument('--destory_method', default='random', type=str, help='The destory method, e.g. llm, random, heuristic')
    parser.add_argument('--repair_method', default='greedy', type=str, help='The repair method, e.g. llm, solver, heuristic')
    parser.add_argument('--problem_category', default='tsp', type=str, help='The problem category, e.g. tsp, cvrp, knapsack, bin_pack, assignment')
    parser.add_argument('--time_limit', default=0, type=int, help='The time limit for each task')
    parser.add_argument('--resume_epoch', type=int, help='The epoch to resume from')
    parser.add_argument('--num_decimals', type=int, default=0, help='The number of decimals to round the coordinates')
    parser.add_argument('--model', default='gpt-4o', type=str, help='The model to use, e.g. gpt-4o, gpt-3.5-turbo, gpt-4.1')
    parser.add_argument('--use_ci', default=0, type=int)
    parser.add_argument('--temp', default=1.0, type=float, help='The initial temperature of acceptance function')
    parser.add_argument('--cooling_rate', default=0.9, type=float, help='The cooling rate of acceptance function')
    parser.add_argument('--opt_val', default=0.0, type=float, help='The optimal value of the problem')

    return parser.parse_args()


def run():
    assert args.problem_category in ['tsp', 'cvrp', 'knapsack', 'bin_pack', 'assignment'], f"Problem category {args.problem_category} not supported"
    assert args.destory_method in ['llm', 'random', 'heuristic'], f"Destory method {args.destory_method} not supported"
    assert args.repair_method in ['llm', 'solver', 'heuristic'], f"Repair method {args.repair_method} not supported"

    logger = setup_logging(args.input_file, args.task, args.reviser_size)

    solve(args.input_file, logger, problem_category=args.problem_category, task_id = args.task, num_iter=args.iterations, init_temperature=args.temp, cooling_rate=args.cooling_rate, reviser_size=args.reviser_size, num_decimals=args.num_decimals, resume_epoch=args.resume_epoch, destory_method=args.destory_method, repair_method=args.repair_method, model=args.model, opt_val=args.opt_val, exp_name=f"destory_{args.destory_method}+repair_{args.repair_method}", use_ci=args.use_ci)


if __name__ == '__main__':

    args = parse_args()
    run()

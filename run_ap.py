import subprocess
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment Problem Solver')
    parser.add_argument('--destory_method', choices=['random', 'llm', 'heuristic'], default='llm', help='The destory method to use')
    parser.add_argument('--repair_method', choices=['llm', 'solver', 'heuristic'], default='llm', help='The repair method to use')
    parser.add_argument('--model', choices=['gpt-4o', 'gpt-4.1', 'o3', 'deepseek-reasoner'], default='gpt-4.1')
    parser.add_argument('--use_ci', choices=['0', '1'], default='0')
    args = parser.parse_args()


    destory_method = args.destory_method
    repair_method = args.repair_method # 'solver', 'llm', 'heuristic'

    ROOT_PATH = 'instances/assignment/large'

    os.makedirs('experiments/assignment', exist_ok=True)

    instances = os.listdir(ROOT_PATH)

    for name in instances:
        if os.path.isfile(f"experiments/destory_{destory_method}+repair_{repair_method}/assignment/{args.model}_{args.use_ci}/{name.split('.')[0]}.sol"):
            print(f"{name} already finished!")
            continue
        print("==================")
        print(f"Start process {name}")
        print("==================")

        input_file = os.path.join(ROOT_PATH, name)

        name = name.split('.')[0]
        try:
            result = subprocess.run(
                [
                    "python",
                    "main.py",
                    "-r", "20",
                    "-n", "100",
                    "-i", input_file,
                    "--model", args.model,
                    "--use_ci", args.use_ci,
                    "--temp", "0.5",
                    "--num_decimals", "0",
                    "--problem_category", "assignment",
                    "--destory_method", destory_method,
                    "--repair_method", repair_method,
                ], timeout=3600,   # in seconds
            )

        except subprocess.TimeoutExpired:
            print("The subprocess exceeded the timeout limit.")
        except subprocess.SubprocessError as e:
            print(f"❌ Subprocess error for {name}: {e}")


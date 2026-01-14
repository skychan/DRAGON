import subprocess
# import pandas as pd
import os
import argparse
import re



def read_opt_val(file_name):
    with open(file_name, "r") as f:
        text = f.read()

    # Updated regex to capture scientific notation
    matches = re.findall(r"Cost\s+([0-9.+-eE]+)", text)
    if matches:
        try:
            cost = float(matches[-1])
            print("Optimal value:", cost)
        except ValueError:
            print("Matched value could not be converted to float:", matches[-1])
            cost = None
    else:
        print("No Optimal value found")
        cost = None
    return cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVRP Problem Solver')
    parser.add_argument('--destory_method', choices=['random', 'llm', 'heuristic'], default='random', help='The destory method to use')
    parser.add_argument('--repair_method', choices=['llm', 'solver', 'heuristic'], default='solver', help='The repair method to use')
    args = parser.parse_args()

    destory_method = args.destory_method
    repair_method = args.repair_method # 'solver', 'llm', 'heuristic'

    os.makedirs(f"experiments/destory_{destory_method}+repair_{repair_method}/cvrp", exist_ok=True)

    # "-i", "instances/CVRP/Vrp-Set-X/X/X-n101-k25.vrp",
    # ROOT_PATH = 'instances/CVRP/Vrp-Set-X/X/'
    ROOT_PATH = 'instances/CVRP/Vrp-Set-XML'
    # uchoa_benchmark_selection = [
    #     "X-n101-k25",    # 100 customers, clustered
    #     "X-n204-k19",    # 203 customers, mixed distribution
    #     "X-n303-k21",    # 302 customers, random
    #     "X-n420-k130",    # 419 customers, large mixed
    #     "X-n1001-k43"    # 1000 customers (largest in benchmark)
    # ]

    uchoa_benchmark_selection = []
    with open('vrp_samples_xml.txt', 'r') as f:
        for line in f.readlines():
            uchoa_benchmark_selection.append(line.strip())

    for name in uchoa_benchmark_selection:
        if os.path.isfile(f"experiments/destory_{destory_method}+repair_{repair_method}/cvrp/{name}.sol"):
            print(f"{name} already finished!")
            continue
        # if name != 'X-n110-k13':
        #     continue
        # get the optimal value
        opt_val = read_opt_val(os.path.join(ROOT_PATH, f"{name}.vrp.sol"))

        print("==================")
        print(f"Start process {name}, optimal value: {opt_val}")
        print("==================")

        input_file = os.path.join(ROOT_PATH, f"{name}.vrp")

        # '''
        try:
            subprocess.run(
                [
                    "python",
                    "main.py",
                    "-i", input_file,
                    "-r", "20",
                    "-n", "100",
                    "--opt_val", f"{opt_val}",
                    "--temp", "0.5",
                    "--num_decimals", "0",
                    "--problem_category", "cvrp",
                    "--destory_method", destory_method,
                    "--repair_method", repair_method,
                ], timeout=3600   # in seconds
            )
        except subprocess.TimeoutExpired:
            print("The subprocess exceeded the timeout limit.")
        except subprocess.SubprocessError as e:
            print(f"❌ Subprocess error for {name}: {e}")

        # '''

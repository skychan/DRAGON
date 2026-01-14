import subprocess
import pandas as pd
import os
import argparse


df_tsp = pd.read_csv('instances/tsplib/meta_info.csv')
df_tsp = df_tsp[df_tsp['EDGE_WEIGHT_TYPE'] == 'EUC_2D']


parser = argparse.ArgumentParser(description='TSP Problem Solver')
parser.add_argument('--destory_method', choices=['random', 'llm', 'heuristic'], default='random', help='The destory method to use')
parser.add_argument('--repair_method', choices=['llm', 'solver', 'heuristic'], default='solver', help='The repair method to use')
parser.add_argument('--model', choices=['gpt-4o', 'gpt-4.1', 'o3', 'deepseek-reasoner'])
parser.add_argument('--use_ci', choices=['0', '1'])
args = parser.parse_args()

destory_method = args.destory_method
repair_method = args.repair_method # 'solver', 'llm', 'heuristic'

os.makedirs('experiments/tsp', exist_ok=True)
for name, opt_val in df_tsp[['NAME', 'bound']].values:
    if os.path.isfile(f"experiments/destory_{destory_method}+repair_{repair_method}/tsp/{args.model}_{args.use_ci}/{name}.sol"):
        print(f"{name} already finished!")
        continue

    print("==================")
    print(f"Start process {name}")
    print("==================")

    try:
        input_file = f"instances/tsplib/{name}.npy" # main(f"tsplib/{name}.npy")
        subprocess.run(
            [
                "python",
                "main.py",
                "-i", input_file,
                "-r", "20",
                "-n", "100",
                "--opt_val", f"{opt_val}",
                "--model", args.model,
                "--use_ci", args.use_ci,
                "--temp", "0.5",
                "--num_decimals", "0",
                "--problem_category", "tsp",
                "--destory_method", destory_method,
                "--repair_method", repair_method,
            ], timeout=3600   # in seconds
        )
    except subprocess.TimeoutExpired:
        print("The subprocess exceeded the timeout limit.")
    except subprocess.SubprocessError as e:
        print(f"❌ Subprocess error for {name}: {e}")

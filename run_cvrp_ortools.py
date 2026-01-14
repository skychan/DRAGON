import subprocess
import pandas as pd
import os
import argparse
import re
import time


def read_opt_val(file_name):
    with open(file_name, "r") as f:
        text = f.read()

    matches = re.findall(r"Cost\s+([0-9.]+)", text)
    if matches:
        cost = float(matches[-1])  # or int(matches[-1]) if always integer
        print("Optimal value:", cost)
    else:
        print("No Optimal value found")
    return cost


if __name__ == '__main__':

    # "-i", "instances/CVRP/Vrp-Set-X/X/X-n101-k25.vrp",
    ROOT_PATH = 'instances/CVRP/Vrp-Set-X/X/'
    # uchoa_benchmark_selection = [
    #     "X-n101-k25",    # 100 customers, clustered
    #     "X-n204-k19",    # 203 customers, mixed distribution
    #     "X-n303-k21",    # 302 customers, random
    #     "X-n420-k130",    # 419 customers, large mixed
    #     "X-n1001-k43"    # 1000 customers (largest in benchmark)
    # ]

    uchoa_benchmark_selection = []
    with open('vrp_samples.txt', 'r') as f:
        for line in f.readlines():
            uchoa_benchmark_selection.append(line.strip())

    csv_path = 'cvrp_orools_gap_time.csv'
    df = pd.DataFrame({'name': [], 'opt_val': [], 'obj': [], 'gap(%)': [], 'infer_time(sec)': []})
    df.to_csv(csv_path, index=False, header=True, mode='w')

    for name in uchoa_benchmark_selection:
        # if os.path.isfile(f"experiments/destory_{destory_method}+repair_{repair_method}/cvrp/{name}.sol"):
        #     print(f"{name} already finished!")
        #     continue
        # if name != 'X-n110-k13':
        #     continue
        # get the optimal value
        opt_val = read_opt_val(os.path.join(ROOT_PATH, f"{name}.sol"))

        print("==================")
        print(f"Start process {name}, optimal value: {opt_val}")
        print("==================")

        input_file = os.path.join(ROOT_PATH, f"{name}.vrp")

    # try:
    #     result = subprocess.run(
    #         ["python", "main.py"],
    #         capture_output=True,
    #         text=True,  # Ensures stdout/stderr are strings
    #         timeout=timeout_sec,
    #         check=True  # Raises CalledProcessError on non-zero exit
    #     )
    #     # Try parsing the output to float
    #     value = float(result.stdout.strip())
    #     return value
    # except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
    #     print(f"Error: {e}")
    #     return -1

        # '''
        start_time = time.time()
        try:
            result = subprocess.run(
                [
                    "python",
                    "run_ortools.py",
                    "-i", input_file,
                    "--time_limit", "3600",
                ], timeout=3630,   # in seconds
                capture_output=True,
                text=True,  # Ensures stdout/stderr are strings
            )
            match = re.search(r"Total distance\s*=\s*([0-9]*\.?[0-9]+)", result.stdout)
            if match:
                obj_val = float(match.group(1))
            else:
                obj_val = float('inf')
            gap = (obj_val - opt_val) / opt_val
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'opt_val': [opt_val], 'obj': [f"{obj_val:.2f}"], 'gap(%)': [f"{gap*100:.3f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        except subprocess.TimeoutExpired:
            print("The subprocess exceeded the timeout limit.")
            obj_val = float('inf')
            gap = float('inf')
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'opt_val': [opt_val], 'obj': [f"{obj_val:.2f}"], 'gap(%)': [f"{gap*100:.3f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        except subprocess.SubprocessError as e:
            print(f"❌ Subprocess error for {name}: {e}")
            obj_val = float('inf')
            gap = float('inf')
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'opt_val': [opt_val], 'obj': [f"{obj_val:.2f}"], 'gap(%)': [f"{gap*100:.3f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        # '''

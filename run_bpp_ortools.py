import subprocess
import pandas as pd
import os
import re
import time



if __name__ == '__main__':


    ROOT_PATH = 'instances/bin_packing/Weibull 5k'

    csv_path = 'bpp_solver_value_time.csv'
    df = pd.DataFrame({'name': [], 'obj': [], 'infer_time(sec)': []})
    df.to_csv(csv_path, index=False, header=True, mode='w')
    df = pd.read_csv(csv_path)

    instances = os.listdir(ROOT_PATH)

    for name in instances:
        if name.split('.')[0] in df['name'].tolist():
            print(f"{name} already finished")
            continue

        print("==================")
        print(f"Start process {name}")
        print("==================")

        input_file = os.path.join(ROOT_PATH, name)

        name = name.split('.')[0]
        start_time = time.time()
        try:
            result = subprocess.run(
                [
                    "python",
                    "solver/bpp.py",
                    "-i", input_file,
                    "--time_limit", "3600",
                ], timeout=3630,   # in seconds
                capture_output=True,
                text=True,  # Ensures stdout/stderr are strings
            )
            print(result.stdout)
            print(result.stderr)
            match = re.search(r"Objective value\s*:\s*([0-9]*\.?[0-9]+)", result.stdout)
            if match:
                obj_val = float(match.group(1))
            else:
                obj_val = float('inf')

            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'obj': [f"{obj_val:.2f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        except subprocess.TimeoutExpired:
            print("The subprocess exceeded the timeout limit.")
            obj_val = float('inf')
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'obj': [f"{obj_val:.2f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        except subprocess.SubprocessError as e:
            print(f"❌ Subprocess error for {name}: {e}")
            obj_val = float('inf')
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'obj': [f"{obj_val:.2f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        # '''

import subprocess
import pandas as pd
import os
import re
import time



if __name__ == '__main__':


    ROOT_PATH = 'instances/knapsack/multi_knapsack_large'

    csv_path = 'mkp_orools_value_time.csv'
    # df = pd.DataFrame({'name': [], 'obj': [], 'infer_time(sec)': []})
    # df.to_csv(csv_path, index=False, header=True, mode='w')
    df = pd.read_csv(csv_path)

    instances = os.listdir(ROOT_PATH)

    names = [
        # 'mkp_393_5_0.json',
        # 'mkp_3594_20_0.json',
        # 'mkp_3851_20_0.json',
        # 'mkp_6870_50_0.json',
        # 'mkp_6922_50_0.json',
        # 'mkp_8864_50_0.json',
        # 'mkp_565_10_0.json',
        #'mkp_10729_80_0.json',
        #'mkp_12397_50_0.json',
        'mkp_30018_100_0.json',
        'mkp_74591_100_0.json',
        'mkp_134493_100_0.json',
        'mkp_293693_100_0.json',
        ]

    for name in names:
        if name.split('.')[0] in df['name'].tolist():
            print(f"{name} already finished")
            continue

        # if name.split('.')[0] in ['mkp_393_5_0', 'mkp_3851_20_0', 'mkp_3594_20_0']:
        #     continue

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
                    "solver/mkp.py",
                    "-i", input_file,
                    "--num_worker", "1",
                    "--time_limit", "3600",
                ], timeout=4000,   # in seconds
                capture_output=True,
                text=True,  # Ensures stdout/stderr are strings
            )
            with open(f'mkp_{name}_out.txt', 'w') as f:
                f.write(result.stdout)
            with open(f'mkp_{name}_err.txt', 'w') as f:
                f.write(result.stderr)
            match = re.search(r"Objective value\s*:\s*([0-9]*\.?[0-9]+)", result.stdout)
            if match:
                obj_val = float(match.group(1))
            else:
                obj_val = -1
            match = re.search(r"Relative gap\s*:\s*([0-9]*\.?[0-9]+)", result.stdout)
            if match:
                gap = float(match.group(1))
            else:
                gap = -1
            match = re.search(r"Bound\s*:\s*([0-9]*\.?[0-9]+)", result.stdout)
            if match:
                bound = float(match.group(1))
            else:
                bound = -1

            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'obj': [f"{obj_val}"], 'bound': [f"{bound}"], 'gap': [f"{gap:.3f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        except subprocess.TimeoutExpired:
            print("The subprocess exceeded the timeout limit.")
            obj_val = -1
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'obj': [f"{obj_val}"], 'bound': [f"{bound}"], 'gap': [f"{gap:.3f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        except subprocess.SubprocessError as e:
            print(f"❌ Subprocess error for {name}: {e}")
            obj_val = -1
            spend = time.time() - start_time
            df = pd.DataFrame({'name': [name], 'obj': [f"{obj_val}"], 'bound': [f"{bound}"], 'gap': [f"{gap:.3f}"], 'infer_time(sec)': [spend]})
            df.to_csv(csv_path, index=False, header=False, mode='a')

        # '''

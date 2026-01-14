import numpy as np
import os
import json

class Assignment():
    def __init__(self, name, max_interval, num_workers, task_id):
        self.name = name
        self.max_interval = max_interval
        self.num_workers = num_workers
        self.task_id = task_id

    def generate_instance(self):
        self.costs = np.random.randint(low=1, high=self.max_interval, size=(self.num_workers, self.num_workers)).tolist()
        return self.costs

    def save_instance(self, dest_folder, filename=None):
        costs = self.generate_instance()
        os.makedirs(dest_folder, exist_ok=True)
        if filename is None:
            filename = f"{self.name.split('_')[0]}_{self.num_workers}_{self.task_id}.json"
        file_path = os.path.join(dest_folder, filename)
        data = {
            "name": self.name,
            "num_workers": self.num_workers,
            "task_id": self.task_id,
            "costs": costs
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Assignment instance saved to {file_path}")

if __name__ == "__main__":
    # Example settings
    name_sets = ["large"]

    size_settings = [
        (5, 10, 1),
        (10, 50, 2),
        (50, 100, 2),
    ]

    max_interval = 100
    dest_folder = "instances/assignment"
    num_tasks_per_setting = 3  # how many instances per setting

    for ns in name_sets:
        for start, end, count in size_settings:
            task_id = 0
            for _ in range(count):
                num_workers = np.random.randint(start, end)
                assignment = Assignment(
                    name=f"{ns}_{num_workers}_{task_id}",
                    max_interval=max_interval,
                    num_workers=num_workers,
                    task_id=task_id
                )
                assignment.save_instance(dest_folder=os.path.join(dest_folder, ns))
                task_id += 1
import numpy as np
import os
import json

class Knapsack():
    def __init__(self, name, max_interval, num_items, capacity, task_id):
        self.name = name
        self.max_interval = max_interval
        self.num_items = num_items
        self.capacity = capacity
        self.task_id = task_id

    def generate_instance(self):
        self.values = np.random.randint(low=1, high=self.max_interval, size=self.num_items).tolist()
        self.weights = np.random.randint(low=1, high=self.max_interval, size=self.num_items).tolist()
        return self.values, self.weights, self.capacity

    def save_instance(self, dest_folder, filename=None):
        values, weights, capacity = self.generate_instance()
        os.makedirs(dest_folder, exist_ok=True)
        if filename is None:
            filename = f"mkp_{self.num_items}_{len(self.capacity)}_{self.task_id}.json"
        file_path = os.path.join(dest_folder, filename)
        data = {
            "name": self.name,
            "capacity": capacity,
            "num_items": self.num_items,
            "values": values,
            "weights": weights
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Knapsack instance saved to {file_path}")

if __name__ == "__main__":
    name_sets = ["multi_knapsack_large"]
    max_interval = 100
    dest_folder = "instances/knapsack"

    # (start, end, count, num_kapsacks)
    # size_settings = [
    #     (100, 500, 1, 5),
    #     (500, 1000, 1, 10),
    #     (1000, 5000, 2, 20),
    #     (5000, 20000, 1, 50),
    # ]

    size_settings = [
        # (5000, 10000, 3, 50),
        # (10000, 20000, 1, 80),
        # (20000, 50000, 1, 100),
        (50000, 100000, 1, 100),
        (100000, 200000, 1, 100),
        (200000, 500000, 1, 100),
    ]

    for ns in name_sets:
        for start, end, count, num_kapsacks in size_settings:
            for _ in range(count):
                num_items = np.random.randint(start, end)
                task_id = 0
                capacities = np.random.randint(low=100, high=500, size=num_kapsacks).tolist()
                # for capacity in capacities:
                knapsack = Knapsack(
                    name=f"{ns}_{num_items}_{task_id}",
                    max_interval=max_interval,
                    num_items=num_items,
                    capacity=capacities,
                    task_id=task_id
                )
                knapsack.save_instance(dest_folder=os.path.join(dest_folder, ns))
                task_id += 1
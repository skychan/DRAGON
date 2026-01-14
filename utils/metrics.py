import time
import tiktoken
import os
import pandas as pd
import numpy as np


class Metrics:
    def __init__(self, model='gpt-4o', use_ci = 1):
        self.start_time = time.time()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.num_api_calls = 0
        self.model_name = model
        # self.encoding = tiktoken.encoding_for_model(model)
        self.tokens_per_message = 3
        self.tokens_per_name = 1
        self.use_ci = use_ci

    def add_input_token_len(self, token_len):
        self.total_input_tokens += token_len

    def add_output_token_len(self, token_len):
        self.total_output_tokens += token_len

    def add_input_tokens(self, message):
        self.total_input_tokens += self.tokens_per_message
        for key, value in message.items():
            self.total_input_tokens += len(self.encoding.encode(value))
            if key == "name":
                self.total_input_tokens += self.tokens_per_name
        self.total_input_tokens += 3  # every reply is primed with 3 tokens

    def add_output_tokens(self, output):
        self.total_output_tokens += len(self.encoding.encode(output))

    def increment_api_calls(self):
        self.num_api_calls += 1

    def get_total_running_time(self):
        return time.time() - self.start_time

    def get_metrics(self):
        return {
            "total_running_time": self.get_total_running_time(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_api_calls": self.num_api_calls,
        }

    def save_csv(self, result_csv_path, tsp_name, min_val, opt_val):
        """
        Save the metrics to a CSV file.

        Args:
            result_csv_path (str): Path to the CSV file.
            tsp_name (str): Name of the TSP file.
            min_dist (float): Minimum distance achieved.
        """
        if min_val is None:
            min_val = np.inf
        metric_result = self.get_metrics()

        # Load existing results if the CSV file exists
        if os.path.exists(result_csv_path):
            results_df = pd.read_csv(result_csv_path)
        else:
            results_df = pd.DataFrame(columns=["name", "opt_val", "obj", "gap(%)", "num_api_calls", "input_tokens", "output_tokens", "running_time(sec)"])

        # Check if the filename already exists in the CSV
        if tsp_name in results_df["name"].values:
            # Update the existing entry
            results_df.loc[results_df["name"] == tsp_name, ["obj", "gap(%)", "num_api_calls", "input_tokens", "output_tokens", "running_time(sec)"]] = [
                min_val,
                "{:.3f}".format((min_val - opt_val) / opt_val * 100),
                metric_result["num_api_calls"],
                metric_result["total_input_tokens"],
                metric_result["total_output_tokens"],
                metric_result["total_running_time"],
            ]
        else:
            # Append a new entry
            new_entry = {
                "name": tsp_name,
                "opt_val": opt_val,
                "obj": min_val,
                "gap(%)": "{:.3f}".format((min_val - opt_val) / opt_val * 100),
                "num_api_calls": metric_result["num_api_calls"],
                "input_tokens": metric_result["total_input_tokens"],
                "output_tokens": metric_result["total_output_tokens"],
                "running_time(sec)": metric_result["total_running_time"],
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_entry])], ignore_index=True)

        # Save the updated results back to the CSV
        results_df.to_csv(result_csv_path, index=False)

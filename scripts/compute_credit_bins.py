import argparse
import json

import numpy as np
import pandas as pd


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--output-path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    train_df = pd.read_json(args.train_data_path, orient="records")
    optimal_credit_bins = {}
    for credit_name in ["barely_true_count",
                        "false_count",
                        "half_true_count",
                        "mostly_true_count",
                        "pants_fire_count"]:
        optimal_credit_bins[credit_name] = list(np.histogram_bin_edges(train_df[credit_name],
                                                                       bins=10))
    with open(args.output_path, "w") as f:
        print(optimal_credit_bins)
        json.dump(optimal_credit_bins, f)

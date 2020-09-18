import argparse
import csv
import json
import os
from typing import Dict
from typing import List

from fake_news.utils.features import normalize_and_clean


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--val-data-path", type=str)
    parser.add_argument("--test-data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()


def read_datapoints(datapath: str) -> List[Dict]:
    with open(datapath) as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=[
            "id",
            "statement_json",
            "label",
            "statement",
            "subject",
            "speaker",
            "speaker_title",
            "state_info",
            "party_affiliation",
            "barely_true_count",
            "false_count",
            "half_true_count",
            "mostly_true_count",
            "pants_fire_count",
            "context",
            "justification"
        ])
        return [row for row in reader]


if __name__ == "__main__":
    args = read_args()
    train_datapoints = read_datapoints(args.train_data_path)
    val_datapoints = read_datapoints(args.val_data_path)
    test_datapoints = read_datapoints(args.test_data_path)
    
    train_datapoints = normalize_and_clean(train_datapoints)
    val_datapoints = normalize_and_clean(val_datapoints)
    test_datapoints = normalize_and_clean(test_datapoints)
    
    with open(os.path.join(args.output_dir, "cleaned_train_data.json"), "w") as f:
        json.dump(train_datapoints, f)
    
    with open(os.path.join(args.output_dir, "cleaned_val_data.json"), "w") as f:
        json.dump(val_datapoints, f)
    
    with open(os.path.join(args.output_dir, "cleaned_test_data.json"), "w") as f:
        json.dump(test_datapoints, f)

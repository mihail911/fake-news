import os
from typing import Dict
from typing import List

import numpy as np
import torch
from transformers import RobertaTokenizerFast

from fake_news.utils.features import Datapoint


class FakeNewsTorchDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict, datapoints: List[Datapoint]):
        self.data = []
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_model_output_path = os.path.join(base_dir, config["model_output_path"])
        tokenizer = RobertaTokenizerFast.from_pretrained(config["tokenizer_path"],
                                                         cache_dir=full_model_output_path,
                                                         padding_side="right")
        for datapoint in datapoints:
            tokenized = tokenizer(datapoint.statement,
                                  padding="max_length",
                                  max_length=config["max_seq_len"],
                                  truncation=True,
                                  return_tensors="np",
                                  return_token_type_ids=True,
                                  return_attention_mask=True,
                                  return_special_tokens_mask=True)
            # Only a single encoding since only a single datapoint tokenized
            self.data.append({
                "ids": tokenized.data["input_ids"].squeeze(),
                "type_ids": tokenized.data["token_type_ids"].squeeze(),
                "attention_mask": tokenized.data["attention_mask"].squeeze(),
                "special_tokens_mask": tokenized.data["special_tokens_mask"].squeeze(),
                "label": np.array(int(datapoint.label))
            })
    
    def __getitem__(self, idx: int):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

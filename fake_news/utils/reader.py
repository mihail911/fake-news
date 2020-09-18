import json
from typing import List

from fake_news.utils.features import Datapoint


def read_json_data(datapath: str) -> List[Datapoint]:
    with open(datapath) as f:
        datapoints = json.load(f)
        return [Datapoint(**point) for point in datapoints]

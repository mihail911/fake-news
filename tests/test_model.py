import numpy as np
import pytest

from fake_news.model.tree_based import RandomForestModel
from fake_news.utils.features import Datapoint


@pytest.fixture
def config():
    return {
        "evaluate": False,
        "model_output_path": "",
        "featurizer_output_path": "",
        "credit_bins_path": "tests/fixtures/optimal_credit_bins.json",
        "params": {}
    }


@pytest.fixture
def sample_datapoints():
    return [
        Datapoint(statement="sample statement 1 asd as",
                  barely_true_count=1,
                  false_count=1,
                  half_true_count=1,
                  mostly_true_count=1,
                  pants_fire_count=1,
                  subject="",
                  speaker="",
                  speaker_title="",
                  state_info="",
                  party_affiliation="",
                  label=True),
        Datapoint(statement="sample statement 2 asfa",
                  barely_true_count=2,
                  false_count=2,
                  half_true_count=2,
                  mostly_true_count=2,
                  pants_fire_count=2,
                  subject="",
                  speaker="",
                  speaker_title="",
                  state_info="",
                  party_affiliation="",
                  label=False),
        Datapoint(statement="sample statement 3 as dfa",
                  barely_true_count=3,
                  false_count=3,
                  half_true_count=3,
                  mostly_true_count=3,
                  pants_fire_count=3,
                  subject="",
                  speaker="",
                  speaker_title="",
                  state_info="",
                  party_affiliation="",
                  label=True)
    ]


def test_rf_overfits_small_dataset(config, sample_datapoints):
    model = RandomForestModel(config=config)
    train_labels = [True, False, True]
    
    model.train(sample_datapoints)
    predicted_labels = np.argmax(model.predict(sample_datapoints), axis=1)
    predicted_labels = list(map(lambda x: bool(x), predicted_labels))
    assert predicted_labels == train_labels


def test_rf_correct_predict_shape(config, sample_datapoints):
    model = RandomForestModel(config=config)
    
    model.train(sample_datapoints)
    predicted_labels = np.argmax(model.predict(sample_datapoints), axis=1)
    
    assert predicted_labels.shape[0] == 3


def test_rf_correct_predict_range(config, sample_datapoints):
    model = RandomForestModel(config=config)
    
    model.train(sample_datapoints)
    predicted_probs = model.predict(sample_datapoints)
    
    assert (predicted_probs <= 1).all()

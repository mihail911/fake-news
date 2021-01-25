import logging
import os
import pickle
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from fake_news.model.base import Model
from fake_news.utils.features import Datapoint
from fake_news.utils.features import TreeFeaturizer

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


class RandomForestModel(Model):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config
        model_cache_path = os.path.join(config["model_output_path"], "model.pkl")
        self.featurizer = TreeFeaturizer(os.path.join(config["featurizer_output_path"],
                                                      "featurizer.pkl"),
                                         config)
        if "evaluate" in config and config["evaluate"] and not os.path.exists(model_cache_path):
            raise ValueError("Model output path does not exist but in `evaluate` mode!")
        if model_cache_path and os.path.exists(model_cache_path):
            LOGGER.info("Loading model from cache...")
            with open(model_cache_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            LOGGER.info("Initializing model from scratch...")
            self.model = RandomForestClassifier(**self.config["params"])
    
    def train(self,
              train_datapoints: List[Datapoint],
              val_datapoints: List[Datapoint] = None,
              cache_featurizer: Optional[bool] = False) -> None:
        self.featurizer.fit(train_datapoints)
        if cache_featurizer:
            feature_names = self.featurizer.get_all_feature_names()
            with open(os.path.join(self.config["model_output_path"],
                                   "feature_names.pkl"), "wb") as f:
                pickle.dump(feature_names, f)
            self.featurizer.save(os.path.join(self.config["featurizer_output_path"],
                                              "featurizer.pkl"))
        train_labels = [datapoint.label for datapoint in train_datapoints]
        LOGGER.info("Featurizing data from scratch...")
        train_features = self.featurizer.featurize(train_datapoints)
        self.model.fit(train_features, train_labels)
    
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        expected_labels = [datapoint.label for datapoint in eval_datapoints]
        predicted_proba = self.predict(eval_datapoints)
        predicted_labels = np.argmax(predicted_proba, axis=1)
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        split_prefix = "" if split is None else split
        return {
            f"{split_prefix} f1": f1,
            f"{split_prefix} accuracy": accuracy,
            f"{split_prefix} auc": auc,
            f"{split_prefix} true negative": tn,
            f"{split_prefix} false negative": fn,
            f"{split_prefix} false positive": fp,
            f"{split_prefix} true positive": tp,
        }
    
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        features = self.featurizer.featurize(datapoints)
        return self.model.predict_proba(features)
    
    def get_params(self) -> Dict:
        return self.model.get_params()
    
    def save(self, model_cache_path: str) -> None:
        LOGGER.info("Saving model to disk...")
        # TODO (mihail): Save using joblib
        with open(model_cache_path, "wb") as f:
            pickle.dump(self.model, f)

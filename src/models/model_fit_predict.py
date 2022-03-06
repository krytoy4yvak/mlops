import pickle
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    recall_score,
    accuracy_score,
    precision_score,
)

from enities import TrainingParams

SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestClassifier":

        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame, need_proba: bool = True
) -> List[np.ndarray]:
    predicts = model.predict(features)
    if need_proba:
        predicts_proba = model.predict_proba(features)
        return [predicts, predicts_proba[:, 1]]
    else:
        return [predicts, np.array()]


def evaluate_model(
    predicts: List[np.ndarray], target: pd.Series, need_proba: bool = True
) -> Dict[str, float]:

    metrics = {
        "f1_score": f1_score(target, predicts[0]),
        "recall_score": recall_score(target, predicts[0]),
        "accuracy_score": accuracy_score(target, predicts[0]),
        "precision_score": precision_score(target, predicts[0]),
    }
    if need_proba:
        metrics.update({"roc_auc_score": roc_auc_score(target, predicts[1])})
    return metrics


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import flwr as fl
except ImportError:  # pragma: no cover - optional dependency
    fl = None

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


def federated_simulation_available() -> bool:
    return fl is not None


def federated_status() -> str:
    if fl is None:
        return "Flower is not installed; federated simulation is stubbed."
    return "Flower is installed; the app can run a local federated SGD demo over patient-partitioned features."


def _build_estimator(random_state: int = 42):
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=random_state,
        max_iter=1,
        tol=None,
    )


def _serialize_model(model: Any) -> np.ndarray:
    raw = pickle.dumps(model)
    return np.frombuffer(raw, dtype=np.uint8)


def _deserialize_model(payload: np.ndarray):
    return pickle.loads(payload.tobytes())


if fl is not None:  # pragma: no cover - optional dependency
    class GlycoGuardClient(fl.client.NumPyClient):
        def __init__(self, X_train, y_train, X_val, y_val, random_state: int = 42):
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.n_features = X_train.shape[1]
            self.classes = np.array([0, 1], dtype=int)
            self.model = _build_estimator(random_state=random_state)
            self._initialize_model()

        def _initialize_model(self):
            if hasattr(self.X_train, "columns"):
                dummy_x = pd.DataFrame(np.zeros((1, self.n_features), dtype=float), columns=self.X_train.columns)
            else:
                dummy_x = np.zeros((1, self.n_features), dtype=float)
            dummy_y = np.array([0], dtype=int)
            self.model.partial_fit(dummy_x, dummy_y, classes=self.classes)

        def get_parameters(self, config):
            return [self.model.coef_.astype(float), self.model.intercept_.astype(float)]

        def set_parameters(self, parameters):
            if parameters:
                self.model.coef_ = np.asarray(parameters[0], dtype=float)
                self.model.intercept_ = np.asarray(parameters[1], dtype=float)
                self.model.classes_ = self.classes
                self.model.t_ = 1.0
                self.model.n_features_in_ = self.n_features

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.partial_fit(self.X_train, self.y_train, classes=self.classes)
            return self.get_parameters(config={}), len(self.X_train), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            proba = self.model.predict_proba(self.X_val)[:, 1]
            auc = roc_auc_score(self.y_val, proba) if len(np.unique(self.y_val)) > 1 else 0.5
            return float(1 - auc), len(self.X_val), {"auc": float(auc)}


def run_local_simulation(patient_partitions, rounds: int = 3, random_state: int = 42) -> dict[str, object]:
    if fl is None:
        return {"status": "unavailable", "detail": federated_status()}

    if not patient_partitions:
        raise ValueError("At least one patient partition is required for federated simulation.")
    clients = [GlycoGuardClient(*partition, random_state=random_state) for partition in patient_partitions]
    global_parameters = clients[0].get_parameters(config={})
    latest_auc = 0.0
    round_metrics: list[dict[str, float]] = []

    for round_idx in range(1, rounds + 1):
        local_results = []
        for client in clients:
            updated_parameters, examples, _ = client.fit(global_parameters, config={"round": round_idx})
            local_results.append((updated_parameters, examples))

        total_examples = sum(examples for _, examples in local_results)
        global_parameters = []
        for param_idx in range(len(local_results[0][0])):
            weighted = sum(parameters[param_idx] * examples for parameters, examples in local_results) / max(total_examples, 1)
            global_parameters.append(weighted)

        round_aucs = []
        for client in clients:
            _, _, metrics = client.evaluate(global_parameters, config={"round": round_idx})
            round_aucs.append(float(metrics["auc"]))
        latest_auc = float(np.mean(round_aucs)) if round_aucs else 0.0
        round_metrics.append({"round": float(round_idx), "auc": latest_auc})

    return {
        "status": "completed",
        "num_clients": len(patient_partitions),
        "rounds": rounds,
        "federated_auc": latest_auc,
        "round_metrics": round_metrics,
    }

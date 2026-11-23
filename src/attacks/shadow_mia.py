"""Phase 3: shadow-model membership inference (Shokri et al., 2017).

The idea that made MIA practical. The attacker can't see inside the target,
but they *can* build their own models that behave like it: train N "shadow"
models on data from the same distribution, with training splits they chose
and therefore know the membership labels for. Now they have a supervised
dataset — (model output on an example) -> (was it a member?) — and can train
an ordinary binary classifier on it. That classifier is the attack, and it
transfers to the target because the target's overfitting leaves the same
signature in its outputs.

Why this beats the Phase 2 threshold baseline: a single loss threshold can
only draw one cut across a one-dimensional score. The attack classifier sees
the whole softmax vector, so it can learn shapes a threshold cannot — e.g.
"confident but with a suspiciously flat runner-up distribution" — and it
learns the *decision rule* from data rather than assuming lower loss is
always more member-like.

Attacker assumptions (the standard Shokri threat model, stated honestly):
the attacker knows the target's architecture and training recipe, and has
auxiliary data from the same distribution. Both are strong. We satisfy them
from ``shadow_pool``, which src.data guarantees is disjoint from the target's
member/non-member evaluation sets — so nothing here leaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .. import data as data_module
from .. import metrics, models, train, utils

logger = logging.getLogger(__name__)


def build_attack_features(stats: dict[str, np.ndarray], sort_probs: bool = True) -> np.ndarray:
    """Turn a model's per-example outputs into attack-classifier features.

    Features: the softmax vector, plus the loss, plus the true-label
    confidence (``num_classes + 2`` columns).

    ``sort_probs=True`` sorts the softmax vector in descending order, which
    discards *which* class was predicted and keeps only the shape of the
    confidence distribution. That makes features comparable across classes so
    one global classifier can pool all the training signal. Set it False for
    per-class attack models, where class identity is already fixed by
    construction and the raw ordering carries usable information (e.g. this
    model systematically confuses cats with dogs on its training set).
    """
    probs = stats["probs"]
    if sort_probs:
        probs = -np.sort(-probs, axis=1)

    return np.column_stack(
        [
            probs,
            stats["loss"],
            stats["confidence_true_label"],
        ]
    )


def _build_classifier(kind: str) -> Pipeline:
    """Construct the attack classifier.

    Wrapped in a StandardScaler pipeline because the feature columns have very
    different ranges (probabilities in [0,1] next to losses that can exceed
    10), which matters for the scale-sensitive models. Tree ensembles ignore
    scaling, so the pipeline is harmless there and keeps one code path.
    """
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0),
    }
    if kind not in classifiers:
        raise ValueError(f"Unknown attack classifier '{kind}'. Available: {sorted(classifiers)}")
    return Pipeline([("scaler", StandardScaler()), ("clf", classifiers[kind])])


@dataclass
class ShadowRun:
    """One shadow model's known-membership outputs."""

    in_stats: dict[str, np.ndarray]
    out_stats: dict[str, np.ndarray]


def train_shadow_models(
    pool: Dataset,
    candidate_indices: np.ndarray,
    num_shadows: int,
    shadow_train_size: int,
    model_config: dict,
    train_config: dict,
    device: torch.device,
    seed: int = 0,
) -> list[ShadowRun]:
    """Train ``num_shadows`` shadow models and collect their in/out outputs.

    Each shadow gets an independent in/out split of ``candidate_indices`` and
    a fresh weight initialization. Shadows are trained with the *same recipe*
    as the target (same architecture and hyperparameters, passed in via
    config) because the attack classifier is learning the signature of that
    recipe's overfitting — train shadows differently and the features shift
    out from under the target.
    """
    shadow_runs: list[ShadowRun] = []
    batch_size = train_config["batch_size"]

    for shadow_index in range(num_shadows):
        # Distinct seed per shadow: different in/out split AND different init.
        shadow_seed = seed + 1000 + shadow_index
        utils.set_seed(shadow_seed)

        in_indices, out_indices = data_module.make_in_out_split(
            candidate_indices, shadow_train_size, seed=shadow_seed
        )
        in_subset = data_module.get_subset(pool, in_indices)
        out_subset = data_module.get_subset(pool, out_indices)

        model = models.build_model(**model_config)
        logger.info("Training shadow model %d/%d (in=%d out=%d)",
                    shadow_index + 1, num_shadows, len(in_indices), len(out_indices))

        model, _ = train.train_model(
            model,
            DataLoader(in_subset, batch_size=batch_size, shuffle=True),
            DataLoader(out_subset, batch_size=batch_size, shuffle=False),
            device,
            epochs=train_config["epochs"],
            lr=train_config["lr"],
            weight_decay=train_config.get("weight_decay", 0.0),
            label_smoothing=train_config.get("label_smoothing", 0.0),
            early_stopping_patience=train_config.get("early_stopping_patience"),
        )

        shadow_runs.append(
            ShadowRun(
                in_stats=train.compute_per_example_stats(model, in_subset, device),
                out_stats=train.compute_per_example_stats(model, out_subset, device),
            )
        )

    return shadow_runs


def _stack_shadow_training_set(
    shadow_runs: list[ShadowRun],
    sort_probs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool every shadow's in/out examples into one supervised dataset.

    Returns (features, membership_labels, true_class_labels).
    """
    feature_blocks, membership_blocks, class_blocks = [], [], []
    for run in shadow_runs:
        for stats, is_member in ((run.in_stats, 1), (run.out_stats, 0)):
            feature_blocks.append(build_attack_features(stats, sort_probs=sort_probs))
            membership_blocks.append(np.full(len(stats["loss"]), is_member, dtype=int))
            class_blocks.append(stats["true_label"])
    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(membership_blocks),
        np.concatenate(class_blocks),
    )


class ShadowAttack:
    """A trained shadow-model membership inference attack.

    ``per_class=True`` fits one classifier per true label (Shokri's original
    design: leakage looks different for an easy class like 'ship' than for a
    confusable one like 'cat'). ``per_class=False`` fits a single classifier
    on class-sorted features, which pools ~10x more training data per model.
    Which wins is an empirical question, so the lab trains both and compares —
    with small shadow counts the global model often wins simply because the
    per-class models are starved of data.
    """

    def __init__(self, classifier_kind: str = "random_forest", per_class: bool = False):
        self.classifier_kind = classifier_kind
        self.per_class = per_class
        self._global_model: Pipeline | None = None
        self._per_class_models: dict[int, Pipeline] = {}

    def fit(self, shadow_runs: list[ShadowRun]) -> "ShadowAttack":
        features, membership, classes = _stack_shadow_training_set(
            shadow_runs, sort_probs=not self.per_class
        )

        # Always fit a global model: for per_class mode it's the fallback for
        # classes with too little data to fit their own classifier.
        self._global_model = _build_classifier(self.classifier_kind).fit(features, membership)

        if self.per_class:
            for class_label in np.unique(classes):
                mask = classes == class_label
                class_membership = membership[mask]
                # Need both labels present, and enough rows to be worth fitting.
                if len(np.unique(class_membership)) < 2 or mask.sum() < 50:
                    logger.warning(
                        "Class %d has too little shadow data (%d rows); falling back to "
                        "the global attack model for it.", class_label, int(mask.sum()),
                    )
                    continue
                self._per_class_models[int(class_label)] = _build_classifier(
                    self.classifier_kind
                ).fit(features[mask], class_membership)

        return self

    def predict_membership_scores(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        """Score each example: higher = more likely a member."""
        if self._global_model is None:
            raise RuntimeError("ShadowAttack.fit() must be called before scoring.")

        features = build_attack_features(stats, sort_probs=not self.per_class)
        if not self.per_class:
            return self._global_model.predict_proba(features)[:, 1]

        scores = np.empty(len(features))
        true_labels = stats["true_label"]
        for class_label in np.unique(true_labels):
            mask = true_labels == class_label
            model = self._per_class_models.get(int(class_label), self._global_model)
            scores[mask] = model.predict_proba(features[mask])[:, 1]
        return scores


def run_shadow_attack(
    shadow_runs: list[ShadowRun],
    member_stats: dict[str, np.ndarray],
    nonmember_stats: dict[str, np.ndarray],
    classifier_kind: str = "random_forest",
    per_class: bool = False,
    with_ci: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit the attack on shadow outputs, evaluate it on the real target.

    Returns (scores, labels, metrics).
    """
    attack = ShadowAttack(classifier_kind=classifier_kind, per_class=per_class).fit(shadow_runs)

    member_scores = attack.predict_membership_scores(member_stats)
    nonmember_scores = attack.predict_membership_scores(nonmember_stats)
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = metrics.build_membership_labels(len(member_scores), len(nonmember_scores))

    # 0.5 is the natural threshold for a calibrated probabilistic classifier,
    # and it was fit only on shadow data, so using it here leaks nothing.
    return scores, labels, metrics.attack_metrics(
        scores, labels, threshold=0.5, with_ci=with_ci
    )

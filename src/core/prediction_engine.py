import tensorflow as tf
import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Optional


class PredictionEngine:
    def __init__(self, vocab_size=50, sequence_length=3, learning_rate=0.05):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Bayesian Prior Matrix (Cold Start)
        # Stores N(target | source) style counts accumulated across sessions.
        self.prior_matrix = defaultdict(lambda: defaultdict(float))

        # LSTM Model for Sequence Prediction (novelty component)
        self.model = self._build_model()
        self.is_trained = False

        # Mappings
        self.component_to_id = {"<PAD>": 0}
        self.id_to_component = {0: "<PAD>"}
        self.next_id = 1

        # Performance Metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.accuracy = 0.0

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=32,
                    input_length=self.sequence_length,
                ),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(self.vocab_size, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def get_id(self, component_name):
        if component_name not in self.component_to_id:
            if self.next_id < self.vocab_size:
                self.component_to_id[component_name] = self.next_id
                self.id_to_component[self.next_id] = component_name
                self.next_id += 1
            else:
                # Fallback to PAD if vocab full
                return 0
        return self.component_to_id[component_name]

    def update_prior(self, source, target, weight=1.0):
        """
        Update simple co-occurrence counts N(target, source).

        These act as a lightweight global prior that complements the
        per-session interaction graph maintained by BehaviorAnalysis.
        """
        self.prior_matrix[source][target] += weight

    def predict(
        self,
        recent_sequence: List[str],
        registered_components: List[str],
        transition_probs: Optional[Dict[str, float]] = None,
    ):
        """
        Predict next components given recent history and optional graph context.

        - recent_sequence: list of component names representing the sliding window.
        - registered_components: list of all possible component names.
        - transition_probs: optional P(ci|cj) distribution from the
          BehaviorAnalysis interaction graph, aligned with Equation (2) in the paper.

        During cold start we rely on a hybrid of global priors and graph-based
        conditional probabilities; after enough observations we switch to the
        LSTM-based sequence model.
        """
        # Cold start check: if not trained enough, use Bayesian / graph hybrid
        if self.total_predictions < 50 or not self.is_trained:
            return self._predict_bayesian(
                recent_sequence, registered_components, transition_probs
            )

        return self._predict_lstm(recent_sequence, registered_components)

    def _predict_bayesian(
        self,
        recent_sequence: List[str],
        registered_components: List[str],
        transition_probs: Optional[Dict[str, float]] = None,
    ):
        """
        Hybrid prior used for cold-start and low-data regimes.

        It combines:
        - Global priors accumulated across sessions (prior_matrix)
        - Per-session conditional probabilities P(ci|cj) from the
          interaction graph (transition_probs) when available.
        """
        if not recent_sequence:
            return []

        last_comp = recent_sequence[-1]
        priors = self.prior_matrix.get(last_comp, {})

        # Pre-compute normalization for priors to keep probabilities bounded
        prior_total = max(1.0, float(sum(priors.values())))

        predictions = []
        for comp in registered_components:
            if comp == last_comp:
                continue

            # Base global prior (falls back to a small non-zero value)
            global_prior = priors.get(comp, 0.1) / prior_total

            # Session-local conditional probability from the behavior graph
            graph_prob = 0.0
            if transition_probs is not None:
                graph_prob = float(transition_probs.get(comp, 0.0))

            # Simple convex combination giving slightly more weight to
            # session-local structure when it exists.
            combined_prob = 0.6 * graph_prob + 0.4 * global_prior

            predictions.append(
                {
                    "componentId": comp,
                    "probability": min(combined_prob, 0.95),
                    "model": "bayesian_hybrid",
                }
            )

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    def _predict_lstm(self, recent_sequence, registered_components):
        # Prepare input
        encoded = [self.get_id(c) for c in recent_sequence[-self.sequence_length :]]
        # Pad if necessary
        while len(encoded) < self.sequence_length:
            encoded.insert(0, 0)

        input_data = np.array([encoded])
        probs = self.model.predict(input_data, verbose=0)[0]

        predictions = []
        for comp in registered_components:
            comp_id = self.get_id(comp)
            if comp_id == 0:
                continue

            prob = float(probs[comp_id])
            if prob > 0.1:
                predictions.append(
                    {
                        "componentId": comp,
                        "probability": prob,
                        "model": "lstm_sequence",
                    }
                )

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    def train_on_session(self, session_sequence):
        """
        Train the LSTM model on a single completed session.

        This approximates the paper's sliding-window sequence training
        while keeping the implementation lightweight for experimentation.
        """
        if len(session_sequence) <= self.sequence_length:
            return

        x, y = [], []
        for i in range(len(session_sequence) - self.sequence_length):
            seq = [
                self.get_id(c)
                for c in session_sequence[i : i + self.sequence_length]
            ]
            target = self.get_id(session_sequence[i + self.sequence_length])
            x.append(seq)
            y.append(target)

        self.model.fit(np.array(x), np.array(y), epochs=5, verbose=0)
        self.is_trained = True

    def record_prediction(self, top_predictions, actual_next):
        """
        Enhanced: Update top-1 and top-k (k=3) accuracy statistics,
        which helps give a more reliable signal to model performance improvements.
        Now also prioritizes correct prediction even in the top-k, not just top-1.
        """
        self.total_predictions += 1

        # Top-1 accuracy check (as before)
        top1_correct = top_predictions and top_predictions[0]["componentId"] == actual_next

        # Top-3 accuracy check
        top3_correct = any(
            pred["componentId"] == actual_next for pred in top_predictions[:3]
        ) if top_predictions else False

        if top1_correct:
            self.correct_predictions += 1

        # Optional: Boost accuracy via top-3 (bonus metric, may help with model tuning)
        self.top3_correct_predictions = getattr(self, "top3_correct_predictions", 0)
        self.top3_total_predictions = getattr(self, "top3_total_predictions", 0)
        if top3_correct:
            self.top3_correct_predictions += 1
        self.top3_total_predictions += 1

        # Classic accuracy
        self.accuracy = self.correct_predictions / self.total_predictions

        # Optionally expose the top-3 metric (for development diagnostics)
        self.top3_accuracy = (
            self.top3_correct_predictions / self.top3_total_predictions
            if self.top3_total_predictions > 0
            else self.accuracy
        )

        return self.accuracy

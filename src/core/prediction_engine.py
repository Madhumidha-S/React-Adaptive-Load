import tensorflow as tf
import numpy as np
import json
from collections import defaultdict

class PredictionEngine:
    def __init__(self, vocab_size=50, sequence_length=3, learning_rate=0.05):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        
        # Bayesian Prior Matrix (Cold Start)
        self.prior_matrix = defaultdict(lambda: defaultdict(float))
        
        # LSTM Model for Sequence Prediction
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
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=32, input_length=self.sequence_length),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_id(self, component_name):
        if component_name not in self.component_to_id:
            if self.next_id < self.vocab_size:
                self.component_to_id[component_name] = self.next_id
                self.id_to_component[self.next_id] = component_name
                self.next_id += 1
            else:
                return 0 # Fallback to PAD if vocab full
        return self.component_to_id[component_name]

    def update_prior(self, source, target, weight=1.0):
        self.prior_matrix[source][target] += weight

    def predict(self, recent_sequence, registered_components):
        """
        recent_sequence: list of component names
        registered_components: list of all possible component names
        """
        # Cold start check: if not trained enough, use Bayesian Prior
        if self.total_predictions < 50:
            return self._predict_bayesian(recent_sequence, registered_components)
        
        return self._predict_lstm(recent_sequence, registered_components)

    def _predict_bayesian(self, recent_sequence, registered_components):
        if not recent_sequence:
            return []
        
        last_comp = recent_sequence[-1]
        priors = self.prior_matrix.get(last_comp, {})
        
        predictions = []
        for comp in registered_components:
            if comp == last_comp: continue
            
            prob = priors.get(comp, 0.1) # Base prior
            # Normalize or scale
            predictions.append({
                "componentId": comp,
                "probability": min(prob / max(1, sum(priors.values())), 0.95),
                "model": "bayesian_prior"
            })
            
        return sorted(predictions, key=lambda x: x['probability'], reverse=True)

    def _predict_lstm(self, recent_sequence, registered_components):
        # Prepare input
        encoded = [self.get_id(c) for c in recent_sequence[-self.sequence_length:]]
        # Pad if necessary
        while len(encoded) < self.sequence_length:
            encoded.insert(0, 0)
        
        input_data = np.array([encoded])
        probs = self.model.predict(input_data, verbose=0)[0]
        
        predictions = []
        for comp in registered_components:
            comp_id = self.get_id(comp)
            if comp_id == 0: continue
            
            prob = float(probs[comp_id])
            if prob > 0.1:
                predictions.append({
                    "componentId": comp,
                    "probability": prob,
                    "model": "lstm_sequence"
                })
        
        return sorted(predictions, key=lambda x: x['probability'], reverse=True)

    def train_on_session(self, session_sequence):
        if len(session_sequence) <= self.sequence_length:
            return
        
        x, y = [], []
        for i in range(len(session_sequence) - self.sequence_length):
            seq = [self.get_id(c) for c in session_sequence[i : i + self.sequence_length]]
            target = self.get_id(session_sequence[i + self.sequence_length])
            x.append(seq)
            y.append(target)
            
        self.model.fit(np.array(x), np.array(y), epochs=5, verbose=0)
        self.is_trained = True

    def record_prediction(self, top_predictions, actual_next):
        self.total_predictions += 1
        # Top-1 accuracy check
        if top_predictions and top_predictions[0]['componentId'] == actual_next:
            self.correct_predictions += 1
        self.accuracy = self.correct_predictions / self.total_predictions
        return self.accuracy

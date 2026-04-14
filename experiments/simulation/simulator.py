import time
import random
from src.core.prediction_engine import PredictionEngine
from src.core.dynamic_loader import DynamicLoader
from src.core.behavior_analysis import BehaviorAnalysis

class SimulationEnvironment:
    def __init__(self, components, enable_novelties=True, battery=1.0, charging=True, downlink=10.0, rtt=50):
        self.components = components
        self.enable_novelties = enable_novelties
        
        # Modules
        self.predictor = PredictionEngine(vocab_size=len(components)+2)
        self.loader = DynamicLoader(energy_aware=enable_novelties, network_aware=True)
        self.behavior = BehaviorAnalysis()
        
        # Initial Context
        self.loader.update_context(battery=battery, charging=charging, downlink=downlink, rtt=rtt)
        
        # Register components
        for c_id, data in components.items():
            self.loader.register_component(c_id, data['size'], data['ux_gain'])
            
        self.battery = battery
        self.downlink = downlink
        self.rtt = rtt

    def run_session(self, session_sequence, trace=False):
        """
        Run a single simulated session through the environment.

        When trace=True, this method additionally records per-step prediction
        details that can be consumed by evaluation utilities (e.g., for
        confusion matrix or error analysis as described in the paper).
        """
        # Reset per-session state
        self.behavior.reset()
        self.loader.reset()

        total_load_time = 0
        total_steps = 0
        trace_events = []

        for i, interaction in enumerate(session_sequence):
            # Support both raw ID strings and dicts with features
            if isinstance(interaction, dict):
                comp_id = interaction["component"]
                dwell_time = interaction.get(
                    "duration", interaction.get("dwell_time", 0)
                )
                # Record behavior with features
                self.behavior.record_interaction(comp_id, dwell_time=dwell_time)
            else:
                comp_id = interaction
                self.behavior.record_interaction(comp_id)

            # 2. Simulate Load Time
            base_time = self._calculate_latency(comp_id)

            # Check if preloaded
            if comp_id in self.loader.loaded_components:
                # Preloaded! Reduced time (90% reduction)
                self.loader.mark_used(comp_id)
                load_time = base_time * 0.1
            else:
                load_time = base_time

            total_load_time += load_time
            total_steps += 1

            # 3. Predict NEXT components
            if i >= 1:
                recent = self.behavior.get_recent_interactions(3)
                registered_ids = list(self.components.keys())

                # Session-local conditional distribution P(ci|cj) from the
                # behavior interaction graph (Equation 2 in the paper).
                transition_probs = {}
                if recent:
                    last_inter = recent[-1]
                    last_comp = last_inter.get("componentId", last_inter) if isinstance(last_inter, dict) else last_inter
                    transition_probs = self.behavior.get_transition_distribution(
                        last_comp
                    )

                # Novelty logic vs Baseline logic
                if self.enable_novelties:
                    predictions = self.predictor.predict(
                        recent, registered_ids, transition_probs=transition_probs
                    )
                    self.loader.process_predictions(predictions)
                else:
                    # Baseline: Simple threshold (Guess.js style).
                    # We still allow the hybrid prior to use transition_probs,
                    # but keep a fixed probability threshold to emulate
                    # non-adaptive strategies described in the paper.
                    predictions = self.predictor._predict_bayesian(
                        recent, registered_ids, transition_probs=transition_probs
                    )
                    # Baseline doesn't have MOP or energy awareness, just threshold > 0.75
                    to_preload = [p for p in predictions if p["probability"] > 0.75]
                    for p in to_preload[:2]:
                        self.loader._simulate_preload(p["componentId"])

                # Update accuracy metrics based on NEXT item (if exists)
                if i < len(session_sequence) - 1:
                    next_interaction = session_sequence[i + 1]
                    next_actual = (
                        next_interaction["component"]
                        if isinstance(next_interaction, dict)
                        else next_interaction
                    )

                    # We re-run predict to get top-1 for the next item
                    preds = self.predictor.predict(
                        recent, registered_ids, transition_probs=transition_probs
                    )
                    acc = self.predictor.record_prediction(preds, next_actual)

                    if trace:
                        top1 = preds[0] if preds else None
                        trace_events.append(
                            {
                                "step": i,
                                "recent": [r.get("componentId", r) if isinstance(r, dict) else r for r in recent],
                                "predicted_top1": top1["componentId"]
                                if top1
                                else None,
                                "predicted_prob": top1["probability"]
                                if top1
                                else None,
                                "actual_next": next_actual,
                                "running_accuracy": acc,
                            }
                        )

                    # Update priors for cold start learning
                    self.predictor.update_prior(recent, next_actual)

        # Post-session training for Transformer if novelty enabled
        if self.enable_novelties and self.predictor.total_predictions > 20:
            self.predictor.train_on_session(self.behavior.history)

        result = {
            "avg_load_time": total_load_time / total_steps,
            "accuracy": self.predictor.accuracy,
            "preloaded_count": self.loader.preloaded_count,
            "wasted_size": self.loader.wasted_bandwidth,
        }
        if trace:
            result["trace"] = trace_events
        return result

    def _calculate_latency(self, component_id):
        comp = self.components.get(component_id)
        if not comp: return 50
        size = comp['size']
        # Latency = RTT + (Size / Bandwidth)
        return self.rtt + ((size * 8) / (self.downlink * 1024) * 1000)

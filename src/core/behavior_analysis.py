from collections import defaultdict


class BehaviorAnalysis:
    def __init__(self):
        # Raw per-interaction history with simple contextual features
        self.history = []
        # Ordered list of component IDs visited in this session
        self.session_sequence = []
        # Per-session transition graph G = (V, E) capturing N(ci, cj) co-occurrence
        # where an edge cj -> ci counts how often ci follows cj (Equation 2 in paper).
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self._last_component_id = None

    def record_interaction(self, component_id, velocity=0, scroll=0, dwell=0, dwell_time=None):
        """
        Record a single interaction with optional behavioral features.

        The paper models user behavior as a graph and uses conditional
        probabilities P(ci|cj) based on co-occurrence counts N(ci, cj).
        Here we maintain those counts online while keeping the original
        lightweight API used by the simulator.
        """
        if dwell_time is not None:
            dwell = dwell_time

        # Update transition graph using the previous component (if any)
        if self._last_component_id is not None:
            self.transition_counts[self._last_component_id][component_id] += 1

        interaction = {
            "componentId": component_id,
            "velocity": velocity,
            "scroll": scroll,
            "dwell": dwell,
        }
        self.history.append(interaction)
        self.session_sequence.append(component_id)
        self._last_component_id = component_id
        return interaction

    def get_recent_sequence(self, length=3):
        """Return the most recent component visit sequence (sliding window)."""
        return self.session_sequence[-length:]

    def get_transition_distribution(self, from_component):
        """
        Compute P(ci | from_component) over the current session graph.

        This directly corresponds to Equation (2) in the paper:
            P(ci|cj) = N(ci, cj) / sum_k N(ck, cj)
        and is used as an additional context-aware signal during cold start.
        """
        counts = self.transition_counts.get(from_component, {})
        total = sum(counts.values())
        if not counts or total == 0:
            return {}
        return {comp_id: count / total for comp_id, count in counts.items()}

    def reset(self):
        """Reset all per-session behavior state and interaction graph."""
        self.history = []
        self.session_sequence = []
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self._last_component_id = None

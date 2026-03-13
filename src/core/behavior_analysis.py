class BehaviorAnalysis:
    def __init__(self):
        self.history = []
        self.session_sequence = []

    def record_interaction(self, component_id, velocity=0, scroll=0, dwell=0, dwell_time=None):
        if dwell_time is not None:
            dwell = dwell_time
        interaction = {
            "componentId": component_id,
            "velocity": velocity,
            "scroll": scroll,
            "dwell": dwell
        }
        self.history.append(interaction)
        self.session_sequence.append(component_id)
        return interaction

    def get_recent_sequence(self, length=3):
        return self.session_sequence[-length:]

    def reset(self):
        self.history = []
        self.session_sequence = []

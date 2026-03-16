class DynamicLoader:
    def __init__(self, energy_aware=True, network_aware=True):
        self.energy_aware = energy_aware
        self.network_aware = network_aware
        
        # Context
        self.battery_level = 1.0 # 0.0 to 1.0
        self.is_charging = True
        self.network_downlink = 10.0 # Mbps
        self.network_rtt = 50 # ms
        
        self.registered_components = {}
        self.loaded_components = set()
        
        # Metrics
        self.preloaded_count = 0
        self.used_preloaded_count = 0
        self.wasted_bandwidth = 0
        self.total_saved_time = 0

    def register_component(self, component_id, size_kb, ux_gain):
        self.registered_components[component_id] = {
            "size": size_kb,
            "ux_gain": ux_gain # 1-10
        }

    def update_context(self, battery=None, charging=None, downlink=None, rtt=None):
        if battery is not None: self.battery_level = battery
        if charging is not None: self.is_charging = charging
        if downlink is not None: self.network_downlink = downlink
        if rtt is not None: self.network_rtt = rtt

    def get_adaptive_threshold(self):
        """
        T = 0.5 + penalty
        """
        threshold = 0.5
        
        # Network penalty
        if self.network_aware:
            if self.network_downlink < 2.0: threshold += 0.2
            if self.network_rtt > 200: threshold += 0.1
            
        # Energy penalty (Novelty)
        if self.energy_aware and not self.is_charging:
            if self.battery_level < 0.2:
                threshold += 0.4 # Very high threshold if battery low
            else:
                threshold += (1.0 - self.battery_level) * 0.2
                
        return min(threshold, 0.95)

    def calculate_mop_score(self, component_id, probability):
        """
        Multi-Objective Optimization Score
        Score = P * (UX / Size)
        """
        comp = self.registered_components.get(component_id)
        if not comp: return 0
        
        size_factor = max(1, comp['size'] / 10.0)
        return probability * (comp['ux_gain'] / size_factor)

    def process_predictions(self, predictions):
        threshold = self.get_adaptive_threshold()
        
        # Filter and score
        candidates = []
        for p in predictions:
            if p['componentId'] in self.loaded_components: continue
            if p['probability'] < threshold: continue
            
            score = self.calculate_mop_score(p['componentId'], p['probability'])
            candidates.append({
                "componentId": p['componentId'],
                "score": score
            })
            
        # Sort by MOP score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit batch size based on context
        batch_size = 3
        if self.network_downlink < 1.0 or (self.energy_aware and self.battery_level < 0.2):
            batch_size = 1
            
        to_load = candidates[:batch_size]
        for c in to_load:
            self._simulate_preload(c['componentId'])
            
        return len(to_load)

    def _simulate_preload(self, component_id):
        if component_id not in self.loaded_components:
            self.loaded_components.add(component_id)
            self.preloaded_count += 1
            comp = self.registered_components.get(component_id)
            if comp:
                self.wasted_bandwidth += comp['size']

    def mark_used(self, component_id):
        if component_id in self.loaded_components:
            self.used_preloaded_count += 1
            comp = self.registered_components.get(component_id)
            if comp:
                # Deduct from wasted if actually used
                self.wasted_bandwidth -= comp['size']
                return True
        return False

    def reset(self):
        self.loaded_components.clear()
        self.preloaded_count = 0
        self.used_preloaded_count = 0
        self.wasted_bandwidth = 0

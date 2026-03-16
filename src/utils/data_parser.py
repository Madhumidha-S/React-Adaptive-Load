import json
import os

class DataParser:
    """
    Utility to parse real-world session data and configurations.
    """
    
    @staticmethod
    def load_scenario_config(file_path):
        """Loads component and pattern configuration from a JSON file."""
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def parse_har_sessions(har_path):
        """
        Parses HAR interactions into session sequences.
        Looks for _chromeInteractions if available.
        """
        if not os.path.exists(har_path):
            return []
            
        with open(har_path, 'r') as f:
            har_data = json.load(f)
            
        sessions = []
        # Basic mapping of element selectors to component IDs
        # This would be expanded in a real scenario
        element_map = {
            '#main-nav-menu': 'Home',
            '.product-card-12': 'Products',
            '#search-input': 'Search',
            '.cart-icon': 'Cart',
            '#checkout-btn': 'Checkout'
        }
        
        # Look for interactions in log/_chromeInteractions
        log = har_data.get('log', {})
        interactions = log.get('_chromeInteractions', [])
        
        if interactions:
            sequence = []
            start_time = interactions[0].get('timestamp', 0)
            
            for inter in interactions:
                target = inter.get('targetElement', '')
                comp_id = None
                for selector, cid in element_map.items():
                    if selector in target:
                        comp_id = cid
                        break
                
                if comp_id:
                    duration = inter.get('timestamp', 0) - start_time
                    sequence.append({
                        'component': comp_id,
                        'dwell_time': duration # This is actually timestamp offset
                    })
            
            if sequence:
                sessions.append(sequence)
                
        return sessions

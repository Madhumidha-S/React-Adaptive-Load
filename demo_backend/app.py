import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.core.prediction_engine import PredictionEngine
from src.core.dynamic_loader import DynamicLoader

app = Flask(__name__)
CORS(app)

# Dummy Components config
COMPONENTS = {
    'Home': {'size': 15, 'ux_gain': 5},
    'Products': {'size': 45, 'ux_gain': 8},
    'Detail': {'size': 85, 'ux_gain': 10},
    'Cart': {'size': 35, 'ux_gain': 9},
    'Checkout': {'size': 65, 'ux_gain': 10}
}
predictor = PredictionEngine(vocab_size=len(COMPONENTS)+2)
loader = DynamicLoader(energy_aware=True, network_aware=True)

for c_id, data in COMPONENTS.items():
    loader.register_component(c_id, data['size'], data['ux_gain'])

# Pre-warm with classic ecommerce paths so prior is immediately confident
paths = [
    ["Home", "Products", "Detail", "Cart", "Checkout"],
    ["Home", "Products", "Detail", "Products", "Detail", "Cart"]
]

def warm_up():
    print("Warming up Python ML Engine...")
    for path in paths:
        for _ in range(50):
            seq = []
            for c in path:
                seq.append({"componentId": c, "dwell": 5000})
                if len(seq) > 1:
                    predictor.update_prior(seq[:-1], seq[-1]["componentId"])
warm_up()


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    recent = data.get('recent_interactions', [])
    battery = data.get('battery', 1.0)
    
    # Update Context for MOP
    loader.update_context(battery=battery)
    
    if not recent:
        return jsonify({"predictions": [], "threshold": loader.get_adaptive_threshold()})
    
    # Clean input for predictor
    clean_recent = []
    for item in recent:
        clean_recent.append({
            "componentId": item['componentId'],
            "dwell": item.get('dwellTime', 0)
        })

    # Run AI Prediction
    preds = predictor.predict(clean_recent, list(COMPONENTS.keys()))
    
    # Process MOP Thresholding
    threshold = loader.get_adaptive_threshold()
    results = []
    
    for p in preds:
        mop_score = loader.calculate_mop_score(p['componentId'], p['probability'])
        
        # Batch restrictor logic (battery saver)
        will_preload = p['probability'] >= threshold
        if battery < 0.20:
            # Under low battery, only preload if it's the absolute explicitly perfect guess (first item AND huge score)
            if len([r for r in results if r['willPreload']]) >= 1:
                will_preload = False
                
        results.append({
            "componentId": p['componentId'],
            "probability": p['probability'],
            "mopScore": mop_score,
            "willPreload": will_preload
        })
    
    return jsonify({
        "predictions": results[:3],
        "threshold": float(threshold),
        "battery": battery
    })

if __name__ == '__main__':
    print("React-Adaptive-Load API is running on port 5001!")
    app.run(port=5001)

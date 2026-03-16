import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.simulation.simulator import SimulationEnvironment
from src.utils.data_parser import DataParser

# Component setup
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sessions', 'ecommerce_config.json')
HAR_PATH = os.path.join(os.path.dirname(__file__), '..', 'reactsmart-performance-analysis', 'data', 'sessions', 'harSample.json')

config = DataParser.load_scenario_config(CONFIG_PATH)
if config:
    COMPONENTS = config['components']
    PATTERNS = config['patterns']
else:
    # Fallback
    COMPONENTS = {
        'Home': {'size': 15, 'ux_gain': 5},
        'Products': {'size': 45, 'ux_gain': 8},
        'Detail': {'size': 85, 'ux_gain': 10},
        'Cart': {'size': 35, 'ux_gain': 9},
        'Checkout': {'size': 65, 'ux_gain': 10}
    }
    PATTERNS = []

def generate_sessions(count=100):
    sessions = []
    # Try to load HAR sessions first
    har_sessions = DataParser.parse_har_sessions(HAR_PATH)
    if har_sessions:
        print(f"📦 Loaded {len(har_sessions)} sessions from HAR")
        sessions.extend(har_sessions)
        count -= len(har_sessions)
    
    # Fill remaining with scenario patterns
    if PATTERNS:
        for _ in range(max(0, count)):
            # Select pattern based on probability
            pattern = random.choices(PATTERNS, weights=[p.get('probability', 0.1) for p in PATTERNS])[0]
            sessions.append(pattern['sequence'])
    else:
        # Simple fallback
        for _ in range(max(0, count)):
            sessions.append(['Home', 'Products', 'Detail', 'Cart', 'Checkout'])
            
    return sessions

def run_benchmark():
    print("🚀 Initializing Python ML Performance Benchmark (Real Data Mode)...")
    sessions = generate_sessions(100)
    
    # Test Scenario: Mobile, Low Battery (Green Computing Test)
    CONTEXT = {
        "battery": 0.15,
        "charging": False,
        "downlink": 3.0, # 3Mbps mobile
        "rtt": 150 # 150ms latency
    }
    
    print(f"\nEnvironment Context: Battery {CONTEXT['battery']*100}%, Downlink {CONTEXT['downlink']}Mbps, RTT {CONTEXT['rtt']}ms")

    # 1. Baseline Run (Decision Tree approx)
    print("\n--- Running Baseline (Basic Thresholds) ---")
    baseline_env = SimulationEnvironment(COMPONENTS, enable_novelties=False, **CONTEXT)
    
    b_latency = 0
    b_acc = 0
    for s in sessions:
        res = baseline_env.run_session(s)
        b_latency += res['avg_load_time']
        b_acc = res['accuracy']
        
    print(f"✅ Baseline Avg Load Time: {b_latency/len(sessions):.2f} ms")
    print(f"✅ Baseline Accuracy: {b_acc * 100:.2f} %")

    # 2. Optimized Run (Novelty: LSTM, MOP, Green)
    print("\n--- Running Optimized (Novelty Engine) ---")
    opt_env = SimulationEnvironment(COMPONENTS, enable_novelties=True, **CONTEXT)
    
    o_latency = 0
    o_acc = 0
    for s in sessions:
        res = opt_env.run_session(s)
        o_latency += res['avg_load_time']
        o_acc = res['accuracy']
        
    print(f"✅ Optimized Avg Load Time: {o_latency/len(sessions):.2f} ms")
    print(f"✅ Optimized Accuracy: {o_acc * 100:.2f} %")
    
    print("\n" + "="*50)
    print("📈 NOVELTY SUMMARY")
    print("="*50)
    print(f"1. Accuracy Jump: {((o_acc - b_acc)*100):.2f}%")
    print(f"2. Note on Load Time: Optimized engine throttled {opt_env.loader.preloaded_count} preloads vs Baseline's {baseline_env.loader.preloaded_count}.")
    print("   This shows Energy-Aware (Green) computing in action for low-battery states.")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()

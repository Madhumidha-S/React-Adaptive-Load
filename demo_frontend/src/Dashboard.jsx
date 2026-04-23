import React from 'react';
import { useML } from './MLContext';
import { Battery, Zap, BrainCircuit, Activity } from 'lucide-react';

export default function Dashboard() {
  const { predictions, battery, setBattery, threshold, preloaded } = useML();

  return (
    <div className="dashboard-panel">
      <div className="dash-header">
        <BrainCircuit className="dash-icon" />
        <h2>React-Adaptive-Load Core</h2>
      </div>

      <div className="card threshold-card">
        <div className="card-header">
            <Activity size={18} />
            <span>MOP Throttle Context</span>
        </div>
        
        <div className="battery-slider">
            <div className="battery-label">
                <Battery size={16} color={battery > 0.2 ? '#4ade80' : '#ef4444'}/>
                <span>Simulated Battery: {(battery * 100).toFixed(0)}%</span>
            </div>
            <input 
                type="range" 
                min="0.05" 
                max="1.0" 
                step="0.05"
                value={battery}
                onChange={(e) => setBattery(parseFloat(e.target.value))}
            />
        </div>

        <div className="threshold-meter">
            <div className="t-label">Dynamic Loader Threshold</div>
            <div className="t-value">{(threshold * 100).toFixed(1)}% req. confidence</div>
            <div className="progress-bg">
                <div className="progress-fill" style={{width: `${threshold * 100}%`, backgroundColor: '#8b5cf6'}}></div>
            </div>
        </div>
      </div>

      <div className="predictions-list">
        <h3>Live AI Predictions</h3>
        {predictions.length === 0 ? (
            <div className="empty-state">Waiting for interaction telemetry...</div>
        ) : (
            predictions.map((p, i) => (
                <div key={i} className={`p-card ${p.willPreload ? 'preloaded' : 'throttled'}`}>
                    <div className="p-header">
                        <span className="p-name">{p.componentId}</span>
                        <span className="p-score">{(p.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="p-status">
                        {p.willPreload ? (
                            <><Zap size={14} /> Background Load Executed</>
                        ) : (
                            <span className="throttled-text">Blocked (Battery Saving)</span>
                        )}
                    </div>
                </div>
            ))
        )}
      </div>

      <div className="stats-box">
         <strong>Active Cache:</strong> {preloaded.size} items ready.
      </div>
    </div>
  );
}

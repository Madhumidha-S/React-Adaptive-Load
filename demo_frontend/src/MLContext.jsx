import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';

const MLContext = createContext({});

export const useML = () => useContext(MLContext);

export const MLProvider = ({ children }) => {
  const [predictions, setPredictions] = useState([]);
  const [battery, setBattery] = useState(1.0); // 100%
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState([]);
  const [preloaded, setPreloaded] = useState(new Set());
  
  const location = useLocation();
  const dwellStartTime = useRef(Date.now());
  const latestInteraction = useRef(null);

  // Translate URL paths to Component IDs
  const getComponentId = (path) => {
    if (path === '/') return 'Home';
    const parts = path.split('/');
    const name = parts[parts.length - 1];
    return name.charAt(0).toUpperCase() + name.slice(1);
  };

  useEffect(() => {
    const currentComponent = getComponentId(location.pathname);

    let currentHistory = [];
    if (latestInteraction.current) {
        latestInteraction.current.dwellTime = Date.now() - dwellStartTime.current;
        setHistory(prev => {
            currentHistory = [...prev, latestInteraction.current].slice(-2); // keep last 2 previous
            return currentHistory;
        });
    }

    latestInteraction.current = { componentId: currentComponent };
    dwellStartTime.current = Date.now();

    // The sequence the AI evaluates should include the page we just landed on
    const aiSequence = [...currentHistory, { componentId: currentComponent }];
    fetchPredictions(aiSequence, battery);

    // Check if the current route was preloaded
    if (preloaded.has(currentComponent)) {
        console.log(`🚀 Accelerated Load! ${currentComponent} was successfully preloaded!`);
    }

  }, [location.pathname]);

  // If user adjusts battery, push new prediction to see MOP throttling activate
  useEffect(() => {
     if (history.length > 0) {
        fetchPredictions(history, battery);
     }
  }, [battery]);

  const fetchPredictions = async (recentInts, currentBattery) => {
    try {
      const res = await fetch('http://localhost:5001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recent_interactions: recentInts,
          battery: currentBattery
        })
      });
      const data = await res.json();
      setPredictions(data.predictions);
      setThreshold(data.threshold);
      
      // Update our internal preloaded cache based on MOP logic
      const newPreloaded = new Set();
      data.predictions.forEach(p => {
          if (p.willPreload) {
              newPreloaded.add(p.componentId);
              // In a real app, this is where we run `import('./Cart')` in the background
          }
      });
      setPreloaded(newPreloaded);

    } catch (e) {
      console.error("Backend offline. Is Flask running?", e);
    }
  };

  return (
    <MLContext.Provider value={{ predictions, battery, setBattery, threshold, preloaded }}>
      {children}
    </MLContext.Provider>
  );
};

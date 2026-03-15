import React, { useCallback, useState } from "react";
import ReactSmartContext from "./ReactSmartContext";

function ReactSmartProvider({ children }) {
  const [interactions, setInteractions] = useState([]);
  const [registeredComponents, setRegisteredComponents] = useState({});
  const [predictions, setPredictions] = useState([]);
  const [currentRoute, setCurrentRoute] = useState("/");

  const recordInteraction = useCallback(
    (type, target, metadata = {}) => {
      const interaction = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        type,
        target,
        metadata,
        route: currentRoute,
        timestamp: Date.now()
      };

      setInteractions((prev) => [...prev, interaction].slice(-50));
    },
    [currentRoute]
  );

  const registerComponent = useCallback((componentId, options = {}) => {
    if (!componentId) return;

    setRegisteredComponents((prev) => ({
      ...prev,
      [componentId]: {
        id: componentId,
        predictionThreshold: options.predictionThreshold ?? 0.75,
        loadDependencies: options.loadDependencies ?? [],
        analyzeInteractions: options.analyzeInteractions ?? true,
        ...options
      }
    }));
  }, []);

  const requestPrediction = useCallback(() => {
    const recentTargets = interactions.slice(-5).map((item) => item.target);
    const componentIds = Object.keys(registeredComponents);

    const rankedPredictions = componentIds
      .filter((id) => !recentTargets.includes(id))
      .slice(0, 3)
      .map((id, index) => ({
        componentId: id,
        confidence: Number((0.9 - index * 0.1).toFixed(2))
      }));

    setPredictions(rankedPredictions);
    return rankedPredictions;
  }, [interactions, registeredComponents]);

  return (
    <ReactSmartContext.Provider
      value={{
        interactions,
        registeredComponents,
        predictions,
        currentRoute,
        recordInteraction,
        registerComponent,
        setCurrentRoute,
        requestPrediction
      }}
    >
      {children}
    </ReactSmartContext.Provider>
  );
}

export default ReactSmartProvider;
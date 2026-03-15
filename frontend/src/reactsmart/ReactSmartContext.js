import { createContext } from "react";

const ReactSmartContext = createContext({
  interactions: [],
  registeredComponents: {},
  predictions: [],
  currentRoute: "/",
  recordInteraction: () => {},
  registerComponent: () => {},
  setCurrentRoute: () => {},
  requestPrediction: () => []
});

export default ReactSmartContext;
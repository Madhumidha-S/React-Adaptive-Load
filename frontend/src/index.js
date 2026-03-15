import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import ReactSmartProvider from "./reactsmart/ReactSmartProvider";

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <ReactSmartProvider>
      <App />
    </ReactSmartProvider>
  </React.StrictMode>
);
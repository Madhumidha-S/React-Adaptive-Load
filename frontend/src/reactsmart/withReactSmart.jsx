import React from "react";
import behaviorCollector from "./behaviorCollector";

function withReactSmart(WrappedComponent) {
  function SmartComponent(props) {
    const handleClick = () => {
      behaviorCollector.record("click", WrappedComponent.name);
    };

    const handleHover = () => {
      behaviorCollector.record("hover", WrappedComponent.name);
    };

    return (
      <div onClick={handleClick} onMouseEnter={handleHover}>
        <WrappedComponent {...props} />
      </div>
    );
  }

  return SmartComponent;
}

export default withReactSmart;
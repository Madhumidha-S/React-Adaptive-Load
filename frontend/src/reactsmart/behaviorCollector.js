const behaviorCollector = {
  interactions: [],

  record(type, target) {
    const interaction = {
      type: type,
      target: target,
      time: Date.now()
    };

    this.interactions.push(interaction);

    if (this.interactions.length > 50) {
      this.interactions.shift();
    }

    console.log("ReactSmart interaction:", interaction);
  },

  getInteractions() {
    return this.interactions;
  }
};

export default behaviorCollector;
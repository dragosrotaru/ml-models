import tf from "npm:@tensorflow/tfjs";
// import "npm:@tensorflow/tfjs-backend-webgpu";

// await tf.setBackend("webgpu");

export class TFNeuralNet {
  public model: tf.Sequential;
  constructor(public params: { inputNodes: number; hiddenNodes: number; outputNodes: number; learningRate: number }) {
    this.model = tf.sequential();

    // Add layers
    // Input layer
    this.model.add(
      tf.layers.dense({
        inputShape: [params.inputNodes],
        units: params.hiddenNodes,
        activation: "sigmoid",
      })
    );
    // Output layer
    this.model.add(
      tf.layers.dense({
        units: params.outputNodes,
        activation: "sigmoid", // or 'softmax' for multi-class classification
      })
    );

    this.model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy", // or 'categoricalCrossentropy' for multi-class
      metrics: ["accuracy"],
    });
  }
}

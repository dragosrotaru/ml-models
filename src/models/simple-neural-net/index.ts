import { Model } from "../../interfaces.ts";
import { NetworkLayer } from "./networkLayer.ts";

/* 
In this model, the Layers manage only their own state,
and the Model passes state to and between the layers.
As such, the interdependency between the layers is
functional.
*/

export class SimpleNeuralNet implements Model {
  private hiddenLayer: NetworkLayer;
  private outputLayer: NetworkLayer;

  constructor(
    public params: {
      inputNodes: number;
      hiddenNodes: number;
      outputNodes: number;
      learningRate: number;
    }
  ) {
    this.hiddenLayer = new NetworkLayer({
      inputNodes: params.inputNodes,
      nodeCount: params.hiddenNodes,
      learningRate: params.learningRate,
    });
    this.outputLayer = new NetworkLayer({
      inputNodes: params.hiddenNodes,
      nodeCount: params.outputNodes,
      learningRate: params.learningRate,
    });
  }

  public feedForward(inputs: number[]): number[] {
    this.hiddenLayer.feedforward(inputs);
    this.outputLayer.feedforward(this.hiddenLayer.outputs);
    return this.outputLayer.outputs;
  }

  public train(inputs: number[], target: number[]): void {
    this.feedForward(inputs);
    this.backpropagateErrors(target);
    this.updateWeightsAndBiases(inputs);
  }

  private updateWeightsAndBiases(inputs: number[]): void {
    this.hiddenLayer.adjustWeightsAndBiases(inputs);
    this.outputLayer.adjustWeightsAndBiases(this.hiddenLayer.outputs);
  }

  private backpropagateErrors(targets: number[]): void {
    this.updateOutputErrors(targets);
    this.updateHiddenErrors();
  }

  private updateOutputErrors(targets: number[]): void {
    this.outputLayer.errors = targets.map((target, i) => target - this.outputLayer.outputs[i]);
  }

  private updateHiddenErrors(): void {
    this.hiddenLayer.errors = Array.from({ length: this.params.hiddenNodes }, (_, i) => {
      return this.outputLayer.errors.reduce((acc, error, j) => {
        // Accumulate the weighted error from each output neuron
        return acc + error * this.outputLayer.weights[j][i]; // weight from hidden neuron i to output neuron j
      }, 0);
    });
  }
}

import { randomArray, sigmoid, sigmoidDerivative } from "../util.ts";

export class NetworkLayer {
  public weights: number[][];
  public biases: number[];
  public outputs: number[];
  public errors: number[];

  constructor(
    public params: {
      inputNodeCount: number;
      nodeCount: number;
      learningRate: number;
    }
  ) {
    this.weights = Array.from({ length: params.nodeCount }, () => randomArray(params.inputNodeCount));
    this.biases = randomArray(params.nodeCount);
    this.outputs = [];
    this.errors = [];
  }

  public feedforward(inputs: number[]) {
    this.outputs = this.weights.map((weightRow, i) =>
      sigmoid(inputs.reduce((acc, input, idx) => acc + input * weightRow[idx], 0) + this.biases[i])
    );
  }

  public calculateGradients(): number[] {
    return this.outputs.map((output, i) => this.errors[i] * sigmoidDerivative(output) * this.params.learningRate);
  }

  public adjustWeightsAndBiases(inputs: number[]): void {
    const gradients = this.calculateGradients();

    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < inputs.length; j++) {
        this.weights[i][j] += gradients[i] * inputs[j];
      }
      this.biases[i] += gradients[i];
    }
  }
}

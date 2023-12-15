import { Model } from "../interfaces.ts";

/** A 3 layer neural network running sequentially on the CPU */
export class SimpleNeuralNet implements Model {
  /* Data Structures */

  private weightsInputHidden: number[][];
  private weightsHiddenOutput: number[][];
  private biasHidden: number[];
  private biasOutput: number[];

  /**
   * Constructs a new instance of the SimpleNeuralNet class.
   *
   * @description Initializes a neural network with a specified number of input, hidden, and output nodes, along with a learning rate.
   * It also initializes the weights and biases for the input-hidden and hidden-output layers with random values.
   *
   * @param {number} inputNodes - The number of nodes in the input layer.
   * @param {number} hiddenNodes - The number of nodes in the hidden layer.
   * @param {number} outputNodes - The number of nodes in the output layer.
   * @param {number} learningRate - The learning rate to be used in training the network.
   */
  constructor(
    public params: {
      inputNodes: number;
      hiddenNodes: number;
      outputNodes: number;
      learningRate: number;
    }
  ) {
    const { inputNodes, hiddenNodes, outputNodes } = params;
    this.weightsInputHidden = Array.from({ length: hiddenNodes }, () =>
      this.randomArray(inputNodes)
    );
    this.weightsHiddenOutput = Array.from({ length: outputNodes }, () =>
      this.randomArray(hiddenNodes)
    );
    this.biasHidden = this.randomArray(hiddenNodes);
    this.biasOutput = this.randomArray(outputNodes);
  }

  /**
   *
   * @returns array of random numbers in the range [-1, 1] of given length
   */
  private randomArray(length: number): number[] {
    return Array.from({ length }, this.randomNumber);
  }
  /**
   *
   * @returns random number in the range [-1, 1]
   */
  private randomNumber(): number {
    return Math.random() * 2 - 1;
  }

  /* Math */

  /**
   * Sigmoid Function
   *
   * @description
   * smooth bounded, differentiable, real function
   * that is defined for all real input values and
   * has a non-negative derivative at each point and exactly one inflection point
   * Latex: sigmoid(x) = 1 / (1 + e^{-x}).
   * Has a steep change around x = 0 and asymptotic behavior towards 0 (for x → -∞) and 1 (for x → ∞)
   *
   * https://en.wikipedia.org/wiki/Sigmoid_function
   *
   * @param {number} x - The input value.
   * @returns {number} The output between 0 and 1.
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Returns the derivative of the sigmoid function
   */
  private sigmoidDerivative(y: number): number {
    return y * (1 - y);
  }

  /**
   * Weighted Sum Calculation
   *
   * @description Computes the weighted sum of inputs * weights and adds a bias.
   * Latex: \sum_{i}(inputs_i \times weights_i) + bias
   *
   * @param {number[]} inputs - Array of input values.
   * @param {number[]} weights - Corresponding weights for each input.
   * @param {number} bias - Bias term to be added to the weighted sum.
   * @returns {number} The computed weighted sum.
   */
  private weightedSum(
    inputs: number[],
    weights: number[],
    bias: number
  ): number {
    return (
      inputs.reduce((acc, input, idx) => acc + input * weights[idx], 0) + bias
    );
  }

  /**
   * Layer Processing in Neural Network
   *
   * @description Processes a layer in a neural network by applying a weighted sum followed by a sigmoid activation function.
   *
   * Latex: output_i = \text{sigmoid}\left(\sum_{j}(inputs_j \times weights_{ij}) + biases_i\right), \quad \forall i
   *
   * @param {number[]} inputs - Array of input values to the layer.
   * @param {number[][]} weights - 2D array of weights, where weights[i][j] is the weight from neuron j in the previous layer to neuron i in the current layer.
   * @param {number[]} biases - Array of bias values for each neuron in the layer.
   * @returns {number[]} The output of the layer after applying the weighted sum and sigmoid activation for each neuron.
   */
  private processLayer(
    inputs: number[],
    weights: number[][],
    biases: number[]
  ): number[] {
    return weights.map((weightRow, i) =>
      this.sigmoid(this.weightedSum(inputs, weightRow, biases[i]))
    );
  }

  /**
   * Adjusts weights and biases based on gradients.
   *
   * @description Updates the weights and biases of a neural network layer using the calculated gradients.
   * This function iteratively adjusts each weight and bias to minimize the error during the training process.
   *
   * @param {number[]} gradients - The gradients calculated for the current layer.
   * @param {number[][]} weights - 2D array of weights of the current layer.
   * @param {number[]} biases - Array of biases of the current layer.
   * @param {number[]} inputs - Outputs from the previous layer, used as inputs to the current layer.
   */
  private adjustWeightsAndBiases(
    gradients: number[],
    weights: number[][],
    biases: number[],
    inputs: number[]
  ): void {
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < inputs.length; j++) {
        weights[i][j] += gradients[i] * inputs[j];
      }
      biases[i] += gradients[i];
    }
  }

  /**
   * Calculates the output errors of the neural network.
   *
   * @description Computes the difference between the expected outputs and the actual outputs of the network.
   * This function is used in the backpropagation step to measure how much the output of the network
   * deviates from the expected results.
   *
   * @param {number[]} expectedOutputs - Array of expected output values.
   * @param {number[]} actualOutputs - Array of actual output values produced by the network.
   * @returns {number[]} An array representing the error for each output node.
   */
  private calculateErrors(
    expectedOutputs: number[],
    actualOutputs: number[]
  ): number[] {
    return expectedOutputs.map((expected, i) => expected - actualOutputs[i]);
  }

  /**
   * Calculates the gradients for backpropagation.
   *
   * @description Computes the gradient of the error with respect to each output,
   * which is used to adjust the weights and biases during the training process.
   * It utilizes the derivative of the sigmoid function as part of the calculation.
   *
   * @param {number[]} outputs - Array of output values from the network.
   * @param {number[]} errors - Array of errors for each output.
   * @param {number} learningRate - The learning rate of the network.
   * @returns {number[]} An array of gradients for each output node.
   */
  private calculateGradients(
    outputs: number[],
    errors: number[],
    learningRate: number
  ): number[] {
    return outputs.map(
      (output, i) => errors[i] * this.sigmoidDerivative(output) * learningRate
    );
  }

  /**
   * Backpropagates the errors from the output layer to the hidden layer.
   *
   * @description Calculates the distribution of errors over the hidden layer neurons,
   * based on the weights and the errors in the output layer.
   * This function is a key part of the backpropagation algorithm, where the error
   * is propagated backward through the network.
   *
   * @param {number[]} outputErrors - Array of errors in the output layer.
   * @param {number[][]} weights - 2D array of weights between the hidden and output layers.
   * @returns {number[]} An array representing the propagated error for each hidden node.
   */
  private backpropagateErrors(
    outputErrors: number[],
    weights: number[][]
  ): number[] {
    return Array.from({ length: this.params.hiddenNodes }, (_, i) => {
      return outputErrors.reduce((acc, error, j) => {
        // Accumulate the weighted error from each output neuron
        return acc + error * weights[j][i];  // weight from hidden neuron i to output neuron j
      }, 0);
    });
  }

  /* High Level Functionality */

  /**
   * Feedforward Operation of Neural Network
   *
   * @description Runs the neural network on an input
   * It processes inputs through successive layers (hidden and output) using the processLayer method.
   * Each layer computes neuron outputs by applying weights and biases, followed by the sigmoid activation function.
   *
   * @param {number[]} inputs - Array of input values to the neural network.
   * @returns {number[]} The outputs array from the final layer of the network.
   */
  public feedForward(inputs: number[]): number[] {
    const hidden = this.processLayer(
      inputs,
      this.weightsInputHidden,
      this.biasHidden
    );
    const output = this.processLayer(
      hidden,
      this.weightsHiddenOutput,
      this.biasOutput
    );
    return output;
  }

  /**
   * Trains the neural network using the provided input and expected output data.
   *
   * @description
   * Performs the training of the neural network through the following steps:
   * 1. Feedforward - Processes the input data through successive layers (hidden and output)
   *    to compute the network's output.
   * 2. Backpropagation - Calculates the error between the expected output and the network's output.
   *    It then backpropagates this error through the network to compute gradients for output and hidden layers.
   * 3. Weight and Bias Adjustment - Adjusts the weights and biases of both hidden-output and input-hidden layers
   *    using the calculated gradients to minimize the error.
   *
   * @param {number[]} inputs - The input data array for the network.
   * @param {number[]} expectedOutputs - The expected output array for the given inputs.
   */
  public train(inputs: number[], expectedOutputs: number[]): void {
    // Feedforward the Network
    const hiddenOutputs = this.processLayer(
      inputs,
      this.weightsInputHidden,
      this.biasHidden
    );
    const outputs = this.processLayer(
      hiddenOutputs,
      this.weightsHiddenOutput,
      this.biasOutput
    );

    // Backpropagate the Errors and calculate Gradients
    const outputErrors = this.calculateErrors(expectedOutputs, outputs);
    const outputGradients = this.calculateGradients(
      outputs,
      outputErrors,
      this.params.learningRate
    );
    
    const hiddenErrors = this.backpropagateErrors(
      outputErrors,
      this.weightsHiddenOutput
      );
    const hiddenGradients = this.calculateGradients(
      hiddenOutputs,
      hiddenErrors,
      this.params.learningRate
    );

    // Adjust the Weights and biases
    this.adjustWeightsAndBiases(
      outputGradients,
      this.weightsHiddenOutput,
      this.biasOutput,
      hiddenOutputs
    );
    this.adjustWeightsAndBiases(
      hiddenGradients,
      this.weightsInputHidden,
      this.biasHidden,
      inputs
    );
  }
}

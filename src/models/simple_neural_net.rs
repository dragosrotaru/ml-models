use rand::prelude::*;
use std::f64::consts::E;

struct LayerWeights {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl LayerWeights {
    /// Initializes a new layer of weights and biases
    fn new(input_nodes: usize, output_nodes: usize) -> Self {
        let mut rng = thread_rng();
        let weights = (0..output_nodes)
            .map(|_| {
                (0..input_nodes)
                    .map(|_| rng.gen_range(-1.0..=1.0))
                    .collect()
            })
            .collect();
        let biases = (0..output_nodes)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        LayerWeights { weights, biases }
    }

    /// Adjusts weights and biases based on gradients.
    fn adjust_weights_and_biases(&mut self, gradients: &[f64], inputs: &[f64]) {
        for (i, &gradient) in gradients.iter().enumerate() {
            for (weight, &input) in self.weights[i].iter_mut().zip(inputs) {
                *weight += gradient * input;
            }
            self.biases[i] += gradient;
        }
    }
}

/// A simple 3-layer neural network.
pub struct SimpleNeuralNet {
    pub input_nodes: usize,
    pub hidden_nodes: usize,
    pub output_nodes: usize,
    pub learning_rate: f64,
    input_to_hidden: LayerWeights,
    hidden_to_output: LayerWeights,
}

impl SimpleNeuralNet {
    /// Constructs a new instance of the SimpleNeuralNet.
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f64,
    ) -> Self {
        SimpleNeuralNet {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            input_to_hidden: LayerWeights::new(input_nodes, hidden_nodes),
            hidden_to_output: LayerWeights::new(hidden_nodes, output_nodes),
        }
    }

    /// Sigmoid activation function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    /// Derivative of the sigmoid function.
    fn sigmoid_derivative(y: f64) -> f64 {
        y * (1.0 - y)
    }

    /// Weighted sum calculation.
    fn weighted_sum(inputs: &[f64], weights: &[f64], bias: f64) -> f64 {
        inputs
            .iter()
            .zip(weights)
            .map(|(&i, &w)| i * w)
            .sum::<f64>()
            + bias
    }

    /// Processes a layer in the neural network.
    fn process_layer(&self, inputs: &[f64], layer: &LayerWeights) -> Vec<f64> {
        layer
            .weights
            .iter()
            .zip(&layer.biases)
            .map(|(weight_row, &bias)| {
                SimpleNeuralNet::sigmoid(SimpleNeuralNet::weighted_sum(inputs, weight_row, bias))
            })
            .collect()
    }

    /// Calculates the output errors of the network.
    fn calculate_errors(expected_outputs: &[f64], actual_outputs: &[f64]) -> Vec<f64> {
        expected_outputs
            .iter()
            .zip(actual_outputs)
            .map(|(&expected, &actual)| expected - actual)
            .collect()
    }

    /// Calculates the gradients for backpropagation.
    fn calculate_gradients(&self, outputs: &[f64], errors: &[f64]) -> Vec<f64> {
        outputs
            .iter()
            .zip(errors)
            .map(|(&output, &error)| {
                error * SimpleNeuralNet::sigmoid_derivative(output) * self.learning_rate
            })
            .collect()
    }

    /// Backpropagates the errors from the output layer to the hidden layer.
    fn backpropagate_errors(
        &self,
        output_errors: &[f64],
        hidden_to_output: &LayerWeights,
    ) -> Vec<f64> {
        (0..self.hidden_nodes)
            .map(|i| {
                output_errors
                    .iter()
                    .zip(&hidden_to_output.weights)
                    .map(|(&error, weights)| error * weights[i])
                    .sum()
            })
            .collect()
    }

    /// Feedforward operation of the neural network.
    pub fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        let hidden_outputs = self.process_layer(inputs, &self.input_to_hidden);
        self.process_layer(&hidden_outputs, &self.hidden_to_output)
    }

    /// Trains the neural network.
    pub fn train(&mut self, inputs: &[f64], expected_outputs: &[f64]) {
        let hidden_outputs = self.process_layer(inputs, &self.input_to_hidden);
        let outputs = self.process_layer(&hidden_outputs, &self.hidden_to_output);

        let output_errors = SimpleNeuralNet::calculate_errors(expected_outputs, &outputs);
        let output_gradients = self.calculate_gradients(&outputs, &output_errors);
        let hidden_errors = self.backpropagate_errors(&output_errors, &self.hidden_to_output);
        let hidden_gradients = self.calculate_gradients(&hidden_outputs, &hidden_errors);

        self.hidden_to_output
            .adjust_weights_and_biases(&output_gradients, &hidden_outputs);
        self.input_to_hidden
            .adjust_weights_and_biases(&hidden_gradients, inputs);
    }
}

mod models;
mod benchmarks;

use models::SimpleNeuralNet;
use benchmarks::mnist::MNISTBenchmark;

fn main() {
    // Hyperparameters
    static INPUT_NODES: usize = 28 * 28; // For MNIST images
    static HIDDEN_NODES: usize = 256;
    static OUTPUT_NODES: usize = 10; // Number of digits (0-9)
    static LEARNING_RATE: f64 = 0.0001;

    println!("initializing model");
    let neural_net = SimpleNeuralNet::new(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE);

    println!("loading benchmark");
    let mut benchmark = MNISTBenchmark::new(neural_net);

    println!("Training");
    benchmark.train(10);

    println!("Testing");
    let result = benchmark.test() * 100.0;

    println!("Result: {:.2}%", result);
}

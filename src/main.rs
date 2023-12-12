mod models;
mod benchmarks;

use models::SimpleNeuralNet;
use benchmarks::mnist::MNISTBenchmark;

fn main() {
    // Initialize the neural network
    static INPUT_NODES: usize = 28 * 28; // For MNIST images
    static HIDDEN_NODES: usize = 128;
    static OUTPUT_NODES: usize = 10; // Number of digits (0-9)
    static LEARNING_RATE: f64 = 0.001;

    let neural_net = SimpleNeuralNet::new(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE);
    println!("Neural net initialized");

    // Initialize the benchmark
    let mut benchmark = MNISTBenchmark::new(neural_net);
    println!("Benchmark data loaded");

    // Train the network
    println!("Training");
    benchmark.train(4); // Train for 4 epochs

    // Test the network
    println!("Testing");
    let result = benchmark.test() * 100.0; // Convert to percentage
    println!("Result: {:.2}%", result);
}

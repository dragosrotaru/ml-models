import { MNISTBenchmark } from "./benchmarks/mnist/benchmark";
import { SimpleNeuralNet } from "./models/simpleNeuralNet";

const neuralNet = new SimpleNeuralNet(28 * 28, 128, 10, 0.001);
console.log("neural net initialized");
const benchmark = new MNISTBenchmark(neuralNet)

console.log("benchmark data loaded");
console.log("training");
benchmark.train(4);
console.log("testing");
const result = benchmark.test() * 100;
console.log(`result: ${result}%`);
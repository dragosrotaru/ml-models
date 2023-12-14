import { MNISTBenchmark } from "./benchmarks/mnist/benchmark.ts";
import { SimpleNeuralNet } from "./models/simpleNeuralNet.ts";

console.log("initializing model");
const neuralNet = new SimpleNeuralNet(28 * 28, 128, 10, 0.001);

console.log("loading benchmark");
const benchmark = new MNISTBenchmark(neuralNet)

console.log("training");
benchmark.train(4);

console.log("testing");
const result = await benchmark.test() * 100;

console.log(`result: ${result}%`);
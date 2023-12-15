import { MNISTBenchmark } from "./benchmarks/mnist/benchmark.ts";
// import { SimpleNeuralNet } from "./models/simpleNeuralNet.ts";
import { WebGPUSimpleNeuralNet } from "./models/webGPUSimpleNeuralNet.ts";

console.log("initializing model");
const neuralNet = await WebGPUSimpleNeuralNet.create({
  inputNodes: 28 * 28,
  hiddenNodes: 128,
  outputNodes: 10,
  learningRate: 0.001,
});

console.log("loading benchmark");
const benchmark = new MNISTBenchmark(neuralNet);

console.log("training");
benchmark.train(1);

console.log("testing");
const result = (await benchmark.test()) * 100;

console.log(`result: ${result}%`);

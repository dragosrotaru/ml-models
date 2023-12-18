import { MNISTBenchmark } from "./benchmarks/mnist/benchmark.ts";
import { SimpleNeuralNet } from "./models/simpleNeuralNet.ts";
import { GPUNeuralNet } from "./models/gpu-neural-net/index.ts";

console.log("initializing model");
/* const neuralNet = new SimpleNeuralNet({
  inputNodes: 28 * 28,
  hiddenNodes: 128,
  outputNodes: 10,
  learningRate: 0.1,
}); */
const neuralNet = await GPUNeuralNet.create({
  inputNodes: 28 * 28,
  hiddenNodes: 128,
  outputNodes: 10,
  learningRate: 0.01,
});

console.log("loading benchmark");
const benchmark = new MNISTBenchmark(neuralNet);

console.log("training");
await benchmark.train(1);

console.log("testing");
const result = (await benchmark.test()) * 100;

console.log(`result: ${result}%`);

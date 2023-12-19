import { NetworkLayer } from "./network-layer.ts";
import { GPUUtil } from "./utils.ts";
import { NetworkLayer as SimpleLayer } from "../simple-neural-net/networkLayer.ts";
import { assert } from "https://deno.land/std@0.209.0/assert/mod.ts";

function assertEqualsApprox(actual: number[], expected: number[], tolerance: number = 0.00001) {
  assert(actual.length === expected.length, "Arrays have different lengths");

  for (let i = 0; i < actual.length; i++) {
    const isApproxEqual = Math.abs(actual[i] - expected[i]) <= tolerance;
    assert(
      isApproxEqual,
      `Index ${i} - Expected ${expected[i]} to be approximately equal to ${actual[i]} within tolerance ${tolerance}`
    );
  }
}

async function getDevice() {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  const device = await adapter?.requestDevice();
  if (!device) throw new Error("no suitable adapter found");
  device.lost.then((e) => console.error(e));
  return device;
}

const device = await getDevice();
const util = new GPUUtil(device);

const params = {
  inputNodeCount: 3,
  nodeCount: 2,
  nextLayerNodeCount: 0,
  learningRate: 1,
};
const input = [0.5, 0.1, -0.3];
const weights = [
  [0.1, 0.2, -0.1],
  [-0.1, 0.1, 0.9],
];
const biases = [0.1, -0.2];
const target = [1, 0];

const simpleLayer = new SimpleLayer(params);
simpleLayer.weights = weights;
simpleLayer.biases = biases;

const layer = new NetworkLayer(device, util, params, util.createAndWriteBuffer(input, "input buffer"));

// Overwriting Random Weights and Biases
const flatWeights = weights.flat();
layer.weights = util.createAndWriteBuffer(flatWeights, "weights buffer");
layer.biases = util.createAndWriteBuffer(biases, "biases buffer");

// binding the buffers
const targetBuffer = util.createAndWriteBuffer(target, "target buffer");
layer.bind(targetBuffer);

Deno.test("feedForward method produces expected output", async () => {
  // Calculate Expected
  simpleLayer.feedforward(input);
  const expected = simpleLayer.outputs;

  // Run layer
  layer.feedForward();
  await device.queue.onSubmittedWorkDone();

  // Read output buffer
  const output = await util.readFromBuffer(layer.outputs);
  assertEqualsApprox(output, expected);
});

Deno.test("calculateErrors method produces expected output", async () => {
  // Calculate Expected
  const expected = target.map((target, i) => target - simpleLayer.outputs[i]);

  // Run layer
  layer.calculateErrors();
  await device.queue.onSubmittedWorkDone();

  // Read output buffer
  const output = await util.readFromBuffer(layer.errors);
  console.log(output, expected);

  assertEqualsApprox(output, expected);
});

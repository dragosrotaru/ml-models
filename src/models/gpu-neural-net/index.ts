import { Model } from "../../interfaces.ts";
import { NetworkLayer } from "./networkLayer.ts";

const DEBUG = "validation"; // out-of-memory, validation

export class GPUNeuralNet implements Model {
  private hiddenLayer: NetworkLayer;
  private outputLayer: NetworkLayer;

  private inputBuffer: GPUBuffer;
  private targetBuffer: GPUBuffer;

  constructor(
    private device: GPUDevice,
    public params: {
      inputNodes: number;
      hiddenNodes: number;
      outputNodes: number;
      learningRate: number;
    }
  ) {
    // Initializing Buffers
    this.inputBuffer = this.device.createBuffer({
      label: "input buffer",
      size: params.inputNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.targetBuffer = this.device.createBuffer({
      label: "expected buffer",
      size: params.outputNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Initializing Layers

    this.hiddenLayer = new NetworkLayer(
      device,
      {
        inputNodeCount: params.inputNodes,
        nodeCount: params.hiddenNodes,
        nextLayerNodeCount: params.outputNodes,
        learningRate: params.learningRate,
      },
      this.inputBuffer
    );

    this.outputLayer = new NetworkLayer(
      device,
      {
        inputNodeCount: params.hiddenNodes,
        nodeCount: params.outputNodes,
        nextLayerNodeCount: 0,
        learningRate: params.learningRate,
      },
      this.hiddenLayer.outputs
    );

    this.hiddenLayer.bind(this.outputLayer);
    this.outputLayer.bind(this.targetBuffer);
  }

  static async create(params: { inputNodes: number; hiddenNodes: number; outputNodes: number; learningRate: number }) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    const device = await adapter?.requestDevice();
    if (!device) throw new Error("no suitable adapter found");
    device.lost.then((e) => console.error(e));
    return new GPUNeuralNet(device, params);
  }

  // Debug
  private async popErrorScope() {
    // pop error scope
    const err = await this.device.popErrorScope();
    if (err) console.error(err);
  }

  // Retrieve Data
  private async readfromBuffer(buffer: GPUBuffer) {
    this.device.pushErrorScope(DEBUG);
    await this.device.queue.onSubmittedWorkDone();

    const readBuffer = this.device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Encode commands for copying buffer to the readable buffer
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
    this.device.queue.submit([commandEncoder.finish()]);

    // Wait for the GPU to finish executing before reading back the data
    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = readBuffer.getMappedRange();

    // Copy data into a Float32Array and return it
    const data = Array.from(new Float32Array(copyArrayBuffer));
    readBuffer.unmap();
    await this.device.queue.onSubmittedWorkDone();
    await this.popErrorScope();
    return data;
  }

  // Feed Forward
  public async feedForward(input: number[]): Promise<number[]> {
    await this.feedForwardDispatch(input);
    return this.readfromBuffer(this.outputLayer.outputs);
  }

  private async feedForwardDispatch(input: number[]): Promise<void> {
    this.device.pushErrorScope(DEBUG);
    this.device.queue.writeBuffer(this.inputBuffer, 0, new Float32Array(input));
    await this.device.queue.onSubmittedWorkDone();
    this.hiddenLayer.feedForward();
    await this.device.queue.onSubmittedWorkDone();
    this.outputLayer.feedForward();
    await this.device.queue.onSubmittedWorkDone();
    await this.popErrorScope();
  }

  // Backpropagation

  public async train(input: number[], target: number[]): Promise<void> {
    // debug and time
    this.device.pushErrorScope(DEBUG);
    const start = performance.now();

    // Feed Forward, clear error buffer, and write target output
    await this.feedForwardDispatch(input);
    this.device.queue.writeBuffer(this.targetBuffer, 0, new Float32Array(target));

    await this.device.queue.onSubmittedWorkDone();

    // log time
    const t1 = performance.now();

    // Backpropagation for Output Layer
    this.outputLayer.calculateErrors();
    this.hiddenLayer.calculateErrors();
    await this.device.queue.onSubmittedWorkDone();

    // log time
    const t2 = performance.now();

    // Backpropagation for Hidden Layer
    this.outputLayer.updateWeightsBiases();
    this.hiddenLayer.updateWeightsBiases();
    await this.device.queue.onSubmittedWorkDone();

    // debug and log time
    const t3 = performance.now();

    const initializeTime = t1 - start;
    const outputLayerTime = t2 - t1;
    const hiddenLayerTime = t3 - t2;
    const totalTime = t3 - start;
    console.log(
      `initialize: ${initializeTime}ms, output: ${outputLayerTime}ms, hidden: ${hiddenLayerTime}ms, total: ${totalTime}ms`
    );

    await this.popErrorScope();
  }
}

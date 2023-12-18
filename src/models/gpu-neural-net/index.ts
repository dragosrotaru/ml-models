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
    this.hiddenLayer = new NetworkLayer(device, {
      inputNodes: params.inputNodes,
      nodeCount: params.hiddenNodes,
      outputNodes: params.outputNodes,
      learningRate: params.learningRate,
    });
    this.outputLayer = new NetworkLayer(device, {
      inputNodes: params.hiddenNodes,
      nodeCount: params.outputNodes,
      outputNodes: params.outputNodes,
      learningRate: params.learningRate,
    });
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
    this.device.popErrorScope();
    return data;
  }

  // Feed Forward
  public async feedForward(input: number[]): Promise<number[]> {
    await this.feedForwardDispatch(input);
    return this.readfromBuffer(this.outputLayer.state);
  }

  private async feedForwardDispatch(input: number[]): Promise<void> {
    this.device.pushErrorScope(DEBUG);
    this.device.queue.writeBuffer(this.inputBuffer, 0, new Float32Array(input));
    this.hiddenLayer.feedForward(this.inputBuffer);
    this.outputLayer.feedForward(this.hiddenLayer.state);
    await this.device.queue.onSubmittedWorkDone();
    await this.popErrorScope();
  }

  // Backpropagation

  public async train(input: number[], expected: number[]): Promise<void> {
    await this.feedForwardDispatch(input);

    this.device.pushErrorScope(DEBUG);
    this.device.queue.writeBuffer(this.targetBuffer, 0, new Float32Array(expected));

    // Backpropagation for Output Layer
    this.outputLayer.backprop(this.hiddenLayer.state, this.targetBuffer);

    // Log all buffers
    await this.device.queue.onSubmittedWorkDone();
    await this.popErrorScope();

    // both errors, and inspect
    /* const inspect = await this.readfromBuffer(this.outputLayer.weights);
    console.log("Inspect:", inspect); */

    this.device.pushErrorScope(DEBUG);

    // Backpropagation for Hidden Layer
    this.hiddenLayer.backprop(this.inputBuffer, this.outputLayer.error);

    // Synchronize after GPU operations
    await this.device.queue.onSubmittedWorkDone();
    await this.popErrorScope();
  }
}

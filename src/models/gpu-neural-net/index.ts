import { Model } from "../../interfaces.ts";
import { GPUUtil } from "./utils.ts";
import { NetworkLayer } from "./network-layer.ts";

export class GPUNeuralNet implements Model {
  private hiddenLayer: NetworkLayer;
  private outputLayer: NetworkLayer;

  private inputBuffer: GPUBuffer;
  private targetBuffer: GPUBuffer;

  constructor(
    private device: GPUDevice,
    private util: GPUUtil,
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
      util,
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
      util,
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
    const util = new GPUUtil(device);
    return new GPUNeuralNet(device, util, params);
  }

  // Feed Forward
  public async feedForward(input: number[]): Promise<number[]> {
    await this.feedForwardDispatch(input);
    return this.util.readFromBuffer(this.outputLayer.outputs);
  }

  private async feedForwardDispatch(input: number[]): Promise<void> {
    this.util.pushErrorScope();
    this.device.queue.writeBuffer(this.inputBuffer, 0, new Float32Array(input));
    await this.device.queue.onSubmittedWorkDone();
    this.hiddenLayer.feedForward();
    await this.device.queue.onSubmittedWorkDone();
    this.outputLayer.feedForward();
    await this.device.queue.onSubmittedWorkDone();
    await this.util.popErrorScope();
  }

  // Backpropagation

  public async train(input: number[], target: number[]): Promise<void> {
    // debug and time
    this.util.pushErrorScope();
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

    await this.util.popErrorScope();
  }
}

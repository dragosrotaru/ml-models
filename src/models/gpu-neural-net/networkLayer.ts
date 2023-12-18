import * as shaders from "./shaders/index.ts";
import { randomNumber } from "../util.ts";

export class NetworkLayer {
  public weights: GPUBuffer;
  public biases: GPUBuffer;
  public outputs: GPUBuffer;
  public errors: GPUBuffer;

  private uniforms: GPUBuffer;

  private bindGroupLayout: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;

  private ffpipeline: GPUComputePipeline;
  private bpPipeline: GPUComputePipeline;
  private updatePipeline: GPUComputePipeline;

  constructor(
    private device: GPUDevice,
    private params: {
      inputNodeCount: number;
      nodeCount: number;
      nextLayerNodeCount: number;
      learningRate: number;
    },
    private inputs: GPUBuffer
  ) {
    // Initializing Buffers
    this.weights = this.createRandomBuffer("weights", params.nodeCount * params.inputNodeCount);
    this.biases = this.createRandomBuffer("bias", params.nodeCount);
    this.outputs = this.createBuffer(params.nodeCount * 4);
    this.errors = this.createBuffer(params.nodeCount * 4);
    this.uniforms = this.createUniformBuffer([
      params.inputNodeCount,
      params.nodeCount,
      params.learningRate,
      params.nextLayerNodeCount,
    ]);

    // Initializing Pipelines
    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: "bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },

        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },

        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },

        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    this.ffpipeline = device.createComputePipeline({
      label: "feedForward pipeline",
      layout: device.createPipelineLayout({
        label: "feedForward pipeline layout",
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: device.createShaderModule({
          label: "feedForward shader",
          code: shaders.feedForward,
        }),
        entryPoint: "main",
      },
    });

    this.bpPipeline = device.createComputePipeline({
      label: "calculate_errors pipeline",
      layout: device.createPipelineLayout({
        label: "calculate_errors pipeline layout",
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: device.createShaderModule({
          label: "calculate_errors shader",
          code: shaders.calculateErrors,
        }),
        entryPoint: "main",
      },
    });

    this.updatePipeline = this.device.createComputePipeline({
      label: "update_weights_biases pipeline",
      layout: this.device.createPipelineLayout({
        label: "update_weights_biases pipeline layout",
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: this.device.createShaderModule({
          label: "update_weights_biases shader",
          code: shaders.updateWeightsAndBiases,
        }),
        entryPoint: "main",
      },
    });
  }

  isLastLayer() {
    return this.params.nextLayerNodeCount === 0;
  }

  public bind(to: NetworkLayer | GPUBuffer) {
    let nextLayerWeights: GPUBuffer;
    let targetsOrNextLayerErrors: GPUBuffer;

    // binding to target layer
    if (to instanceof GPUBuffer) {
      if (!this.isLastLayer()) {
        throw new Error("Cannot bind to a target buffer if this is not the last layer");
      }
      targetsOrNextLayerErrors = to;
      nextLayerWeights = this.createBuffer(4);
    } else {
      if (this.isLastLayer()) {
        throw new Error("Cannot bind to a layer error buffer if this is the last layer");
      }
      nextLayerWeights = to.weights;
      targetsOrNextLayerErrors = to.errors;
    }
    this.bindGroup = this.device.createBindGroup({
      label: "bind group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },

        { binding: 1, resource: { buffer: this.inputs } },
        { binding: 2, resource: { buffer: this.outputs } },
        { binding: 3, resource: { buffer: this.errors } },

        { binding: 4, resource: { buffer: this.weights } },
        { binding: 5, resource: { buffer: this.biases } },

        { binding: 6, resource: { buffer: nextLayerWeights } },
        { binding: 7, resource: { buffer: targetsOrNextLayerErrors } },
      ],
    });
  }

  // Workgroup Size

  private WORKGROUP_SIZE = 128;
  private get workgroupCount() {
    return Math.ceil(this.params.nodeCount / this.WORKGROUP_SIZE);
  }

  // Buffer Creation

  private createBuffer(size: number): GPUBuffer {
    return this.device.createBuffer({
      size, // Assuming each node's output is a single float
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  private createUniformBuffer(values: number[]): GPUBuffer {
    const array = new Float32Array(values);
    const buffer = this.device.createBuffer({
      label: "uniform buffer",
      size: array.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();
    return buffer;
  }

  private createRandomBuffer(label: string, size: number) {
    const array = new Float32Array(size);
    for (let i = 0; i < array.length; ++i) {
      array[i] = randomNumber();
    }

    const buffer = this.device.createBuffer({
      label,
      size: array.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();

    return buffer;
  }

  private pass(pipeline: GPUComputePipeline) {
    if (!this.bindGroup) throw new Error("Bind group not initialized");
    const { device } = this;

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.insertDebugMarker("dispatch");
    pass.dispatchWorkgroups(this.workgroupCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  public feedForward() {
    this.pass(this.ffpipeline);
  }

  public calculateErrors() {
    this.pass(this.bpPipeline);
  }

  public updateWeightsBiases() {
    this.pass(this.updatePipeline);
  }
}

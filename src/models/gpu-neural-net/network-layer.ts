import * as shaders from "./shaders/index.ts";
import { GPUUtil } from "./utils.ts";

export class NetworkLayer {
  private WORKGROUP_SIZE = 128;

  private bindGroupLayout: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;

  private ffpipeline: GPUComputePipeline;
  private bpPipeline: GPUComputePipeline;
  private updatePipeline: GPUComputePipeline;

  private uniforms: GPUBuffer;
  public weights: GPUBuffer;
  public biases: GPUBuffer;
  public outputs: GPUBuffer;
  public errors: GPUBuffer;

  constructor(
    private device: GPUDevice,
    private util: GPUUtil,
    public params: {
      inputNodeCount: number;
      nodeCount: number;
      nextLayerNodeCount: number;
      learningRate: number;
    },
    public inputs: GPUBuffer
  ) {
    // Initializing Buffers
    this.weights = util.createRandomBuffer("weights", params.nodeCount * params.inputNodeCount);
    this.biases = util.createRandomBuffer("bias", params.nodeCount);
    this.outputs = util.createBuffer(params.nodeCount * 4);
    this.errors = util.createBuffer(params.nodeCount * 4);
    this.uniforms = util.createUniformBuffer([
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

  public get workgroupCount(): number {
    return Math.ceil(this.params.nodeCount / this.WORKGROUP_SIZE);
  }

  public get isLastLayer(): boolean {
    return this.params.nextLayerNodeCount === 0;
  }

  public bind(to: NetworkLayer | GPUBuffer) {
    let nextLayerWeights: GPUBuffer;
    let targetsOrNextLayerErrors: GPUBuffer;

    // binding to target layer
    if (to instanceof GPUBuffer) {
      if (!this.isLastLayer) {
        throw new Error("Cannot bind to a target buffer if this is not the last layer");
      }
      targetsOrNextLayerErrors = to;
      nextLayerWeights = this.util.createBuffer(4);
    } else {
      if (this.isLastLayer) {
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

import * as shaders from "./shaders/index.ts";

export class NetworkLayer {
  public weights: GPUBuffer;
  public bias: GPUBuffer;
  public state: GPUBuffer;
  public error: GPUBuffer;

  private uniforms: GPUBuffer;

  constructor(
    private device: GPUDevice,
    private params: { inputNodes: number; nodeCount: number; outputNodes: number; learningRate: number }
  ) {
    this.weights = this.createRandomBuffer("weights", params.nodeCount * params.inputNodes);
    this.bias = this.createRandomBuffer("bias", params.nodeCount);
    this.state = this.createBuffer(params.nodeCount * 4);
    this.error = this.createBuffer(params.nodeCount * 4);
    this.uniforms = this.createUniformBuffer([
      params.inputNodes,
      params.nodeCount,
      params.outputNodes,
      params.learningRate,
    ]);
  }

  // Workgroup Size

  private WORKGROUP_SIZE = 128;
  private get workgroupCount() {
    return Math.ceil(this.params.nodeCount / this.WORKGROUP_SIZE);
  }

  private createBuffer(size: number): GPUBuffer {
    return this.device.createBuffer({
      size, // Assuming each node's output is a single float
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  private createUniformBuffer(values: number[]): GPUBuffer {
    const array = new Float32Array(values);
    const buffer = this.device.createBuffer({
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
      array[i] = this.randomNumber();
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
  private randomNumber(): number {
    return Math.random() * 2 - 1;
  }

  // Buffer Metadata
  private bindGroupLayoutEntry(index: number) {
    return {
      binding: index,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage",
      },
    } as const;
  }

  private bindGroupEntry(index: number, buffer: GPUBuffer) {
    return {
      binding: index,
      resource: {
        buffer,
      },
    } as const;
  }

  public feedForward(inputBuffer: GPUBuffer) {
    const { device } = this;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        this.bindGroupLayoutEntry(1),
        this.bindGroupLayoutEntry(2),
        this.bindGroupLayoutEntry(3),
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
          },
        },
      ],
    });
    const bindGroup = device.createBindGroup({
      label: "feedForward bind group",
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniforms,
          },
        },
        this.bindGroupEntry(1, inputBuffer),
        this.bindGroupEntry(2, this.weights),
        this.bindGroupEntry(3, this.bias),
        this.bindGroupEntry(4, this.state),
      ],
    });

    const pipeline = device.createComputePipeline({
      label: "feedForward pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: device.createShaderModule({
          code: shaders.feedForward,
        }),
        entryPoint: "main",
      },
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.insertDebugMarker("dispatch");
    pass.dispatchWorkgroups(this.workgroupCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  public backprop(input: GPUBuffer, target: GPUBuffer) {
    /*  const inspect = await this.readfromBuffer(uniforms);
    console.log("Inspect:", inspect); */

    // Bind Group Layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Bind Group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniforms } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: this.state } },
        { binding: 4, resource: { buffer: target } },
        { binding: 3, resource: { buffer: this.error } },
        { binding: 5, resource: { buffer: this.weights } },
        { binding: 6, resource: { buffer: this.bias } },
      ],
    });

    // Shader Module and Pipeline
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: this.device.createShaderModule({ code: shaders.backprop }),
        entryPoint: "main",
      },
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroupCount);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }
}

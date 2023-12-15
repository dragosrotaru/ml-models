import * as shaders from "./shaders/index.ts";
import { Model } from "../interfaces.ts";

export class WebGPUSimpleNeuralNet implements Model {
  private weightsInputHiddenBuffer: GPUBuffer;
  private weightsHiddenOutputBuffer: GPUBuffer;
  private biasHiddenBuffer: GPUBuffer;
  private biasOutputBuffer: GPUBuffer;
  private scalarBuffer: GPUBuffer;

  constructor(
    private device: GPUDevice,
    public params: {
      inputNodes: number;
      hiddenNodes: number;
      outputNodes: number;
      learningRate: number;
    }
  ) {
    this.weightsInputHiddenBuffer = this.createRandomBuffer(
      "weightsInputHidden",
      params.hiddenNodes * params.inputNodes
    );
    this.weightsHiddenOutputBuffer = this.createRandomBuffer(
      "weightsHiddenOutput",
      params.outputNodes * params.hiddenNodes
    );
    this.biasHiddenBuffer = this.createRandomBuffer("biasHidden", params.hiddenNodes);
    this.biasOutputBuffer = this.createRandomBuffer("biasOutput", params.outputNodes);
    this.scalarBuffer = this.createScalarBuffer([params.inputNodes, params.outputNodes]);
  }

  static async create(params: { inputNodes: number; hiddenNodes: number; outputNodes: number; learningRate: number }) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    const device = await adapter?.requestDevice();
    if (!device) throw new Error("no suitable adapter found");
    device.lost.then((e) => console.error(e));
    return new WebGPUSimpleNeuralNet(device, params);
  }

  // Buffer Creation

  private createBuffer(size: number): GPUBuffer {
    return this.device.createBuffer({
      size, // Assuming each node's output is a single float
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  private createRandomBuffer(label: string, size: number) {
    const array = new Float32Array(size);
    for (let i = 0; i < array.length; ++i) {
      array[i] = this.randomNumber();
    }

    const buffer = this.device.createBuffer({
      label,
      size: array.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();

    return buffer;
  }

  private randomNumber(): number {
    return Math.random() * 2 - 1;
  }

  private createScalarBuffer(values: number[]): GPUBuffer {
    const array = new Uint32Array(values);
    const buffer = this.device.createBuffer({
      size: array.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(buffer.getMappedRange()).set(array);
    buffer.unmap();
    return buffer;
  }

  // Buffer Metadata
  private bindGroupLayoutEntry(index: number) {
    return {
      binding: index,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage",
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

  private async readFromBuffer(buffer: GPUBuffer, size: number): Promise<number[]> {
    // Create a GPU buffer for reading back in CPU land
    const readBuffer = this.device.createBuffer({
      size: size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Encode commands for copying buffer to the readable buffer
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    this.device.queue.submit([commandEncoder.finish()]);

    // Wait for the GPU to finish executing before reading back the data
    await readBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = readBuffer.getMappedRange();

    // Copy data into a Float32Array and return it
    const data = Array.from(new Float32Array(copyArrayBuffer));
    readBuffer.unmap();
    return data;
  }

  // Workgroup Size

  private WORKGROUP_SIZE = 64;
  private getWorkgroupCount(nodeCount: number) {
    return Math.ceil(nodeCount / this.WORKGROUP_SIZE);
  }

  // Debug

  private async popErrorScope() {
    // pop error scope
    const err = await this.device.popErrorScope();
    if (err) console.error(err);
  }

  // Feed Forward

  public async feedForward(input: number[]): Promise<number[]> {
    this.device.pushErrorScope("validation");
    const { output } = await this.feedForwardBuffer(input);
    const outputArrayBuffer = await this.readFromBuffer(output, this.params.outputNodes * 4);
    // pop error scope
    await this.popErrorScope();
    return Array.from(outputArrayBuffer);
  }

  private async feedForwardBuffer(
    input: number[]
  ): Promise<{ input: GPUBuffer; hidden: GPUBuffer; output: GPUBuffer }> {
    const inputBuffer = this.device.createBuffer({
      label: "input buffer",
      size: input.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(inputBuffer, 0, new Float32Array(input));

    const hiddenOutputBuffer = this.createBuffer(this.params.hiddenNodes * 4);
    const finalOutputBuffer = this.createBuffer(this.params.outputNodes * 4);

    // First Dispatch: Input to Hidden Layer
    this.feedForwardLayer(
      inputBuffer,
      this.weightsInputHiddenBuffer,
      this.biasHiddenBuffer,
      hiddenOutputBuffer,
      this.getWorkgroupCount(this.params.hiddenNodes)
    );

    // Second Dispatch: Hidden Layer to Output Layer
    this.feedForwardLayer(
      hiddenOutputBuffer,
      this.weightsHiddenOutputBuffer,
      this.biasOutputBuffer,
      finalOutputBuffer,
      this.getWorkgroupCount(this.params.outputNodes)
    );

    await this.device.queue.onSubmittedWorkDone();
    return {
      input: inputBuffer,
      hidden: hiddenOutputBuffer,
      output: finalOutputBuffer,
    };
  }

  private feedForwardLayer(
    inputBuffer: GPUBuffer,
    weightBuffer: GPUBuffer,
    biasBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    workgroupCount: number
  ) {
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
            buffer: this.scalarBuffer,
          },
        },
        this.bindGroupEntry(1, inputBuffer),
        this.bindGroupEntry(2, weightBuffer),
        this.bindGroupEntry(3, biasBuffer),
        this.bindGroupEntry(4, outputBuffer),
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
    pass.dispatchWorkgroups(workgroupCount, workgroupCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // Backpropagation

  public async train(input: number[], expected: number[]): Promise<void> {
    const { input: inputBuffer, hidden, output } = await this.feedForwardBuffer(input);

    this.device.pushErrorScope("validation");
    const expectedBuffer = this.device.createBuffer({
      label: "expected buffer",
      size: expected.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(expectedBuffer, 0, new Float32Array(expected));

    // Backpropagation for Output Layer
    const errors = await this.backpropLayer(
      hidden,
      output,
      expectedBuffer,
      this.weightsHiddenOutputBuffer,
      this.biasOutputBuffer,
      {
        inputNodes: this.params.hiddenNodes,
        nodeCount: this.params.outputNodes,
        outputNodes: this.params.outputNodes,
      }
    );

    // Log all buffers
    await this.device.queue.onSubmittedWorkDone();

    // Backpropagation for Hidden Layer
    this.backpropLayer(inputBuffer, hidden, errors, this.weightsInputHiddenBuffer, this.biasHiddenBuffer, {
      inputNodes: this.params.inputNodes,
      nodeCount: this.params.hiddenNodes,
      outputNodes: this.params.outputNodes,
    });

    // Synchronize after GPU operations
    await this.device.queue.onSubmittedWorkDone();

    // pop error scope
    await this.popErrorScope();
  }

  private async backpropLayer(
    input: GPUBuffer,
    output: GPUBuffer,
    outputErrors: GPUBuffer,
    weights: GPUBuffer,
    biases: GPUBuffer,
    counts: {
      inputNodes: number;
      nodeCount: number;
      outputNodes: number;
    }
  ) {
    // Buffers
    const hiddenErrors = this.createBuffer(counts.nodeCount * 4);
    const uniforms = this.createScalarBuffer([
      counts.inputNodes,
      counts.nodeCount,
      counts.outputNodes,
      this.params.learningRate,
    ]);

    // Bind Group Layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Bind Group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: outputErrors } },
        { binding: 4, resource: { buffer: hiddenErrors } },
        { binding: 5, resource: { buffer: weights } },
        { binding: 6, resource: { buffer: biases } },
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
    pass.dispatchWorkgroups(this.getWorkgroupCount(counts.nodeCount), 1, 1);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    //const errorResult = await this.readFromBuffer(output, this.params.outputNodes * 4);

    return hiddenErrors;
  }
}
